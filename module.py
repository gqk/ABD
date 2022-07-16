# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import cl_lite.core as cl
import cl_lite.backbone as bb
from cl_lite.head import DynamicSimpleHead
from cl_lite.mixin import FeatureHookMixin
from cl_lite.nn import freeze
from cl_lite.deep_inversion import GenerativeInversion

from datamodule import DataModule


class Module(FeatureHookMixin, cl.Module):
    datamodule: DataModule
    evaluator_cls = cl.ILEvaluator

    def __init__(
        self,
        base_lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        lr_factor: float = 0.1,
        milestones: List[int] = [80, 120],
        lambda_kd: float = 0.1,
        num_inv_iters: int = 5000,
        inv_lr: float = 0.001,
        inv_tau: float = 1000.0,
        inv_alpha_pr: float = 0.001,
        inv_alpha_rf: float = 50.0,
        inv_resume_from: str = None,
    ):
        """Module of joint project

        Args:
            base_lr: Base learning rate
            momentum: Momentum value for SGD optimizer
            weight_decay: Weight decay value
            lr_factor: Learning rate decay factor
            milestones: Milestones for reducing learning rate
            lambda_kd: the scale factor of weight feature distillation
            num_inv_iters: number of inversion iterations
            inv_lr: inversion learning rate
            inv_tau: temperature of inversion cross entropy loss
            inv_alpha_pr: factor of inversion image prior regularization
            inv_alpha_rf: factor of inversion feature statistics regularization
        """

        super().__init__()
        self.save_hyperparameters()

    def register_losses(self):
        self.register_loss(
            "ce",
            nn.CrossEntropyLoss(),
            ["prediction", "target"],
        )

        if self.model_old is None:
            return

        ratio = self.model_old.head.num_classes / self.head.num_classes

        self.set_loss_factor("ce", 1.0 - ratio)

        weight = torch.ones(self.head.num_classes)
        weight[: self.model_old.head.num_classes] = ratio
        weight[self.model_old.head.num_classes :] = 1 - ratio
        self.register_loss(
            "ft",
            nn.CrossEntropyLoss(weight=weight),
            ["input_ft", "target_ft"],
        )

        self.register_loss(
            "fkd",
            nn.MSELoss(),
            ["input_fkd", "target_fkd"],
            self.hparams.lambda_kd,
        )

    def update_old_model(self):
        model_old = [
            ("backbone", deepcopy(self.backbone)),
            ("head", deepcopy(self.head)),
        ]

        model_old = nn.Sequential(OrderedDict(model_old))
        freeze(model_old)
        self.model_old = model_old.eval()

        self.inversion = GenerativeInversion(
            model=deepcopy(self.model_old),
            dataset=self.datamodule.dataset,
            batch_size=self.datamodule.batch_size,
            max_iters=self.hparams.num_inv_iters,
            lr=self.hparams.inv_lr,
            tau=self.hparams.inv_tau,
            alpha_pr=self.hparams.inv_alpha_pr,
            alpha_rf=self.hparams.inv_alpha_rf,
        )

        self.register_feature_hook("pen", "head.neck")

    def init_setup(self, stage=None):
        if self.datamodule.dataset.startswith("imagenet"):
            self.backbone = bb.resnet.resnet18()
        else:
            self.backbone = bb.resnet_cifar.resnet32()
        self.head = DynamicSimpleHead(num_features=self.backbone.num_features)

        for task_id in range(0, self.datamodule.current_task + 1):
            if task_id > 0 and task_id == self.datamodule.current_task:
                self.update_old_model()  # load from checkpoint
            self.head.append(self.datamodule[task_id].num_classes)

        self.model_old, self.inversion = None, None

    def setup(self, stage=None):
        current_task = self.datamodule.current_task
        resume_from_checkpoint = self.trainer.resume_from_checkpoint

        if current_task == 0 or resume_from_checkpoint is not None:
            self.init_setup(stage)
        else:
            self.update_old_model()
            self.head.append(self.datamodule.num_classes)

        self.register_losses()

        self.print(f"=> Network Overview \n {self}")

    def forward(self, input):
        output = self.backbone(input)
        output = self.head(output)
        return output

    def on_train_start(self):
        super().on_train_start()
        if self.model_old is not None:
            ckpt_path = self.hparams.inv_resume_from
            if ckpt_path is None:
                self.inversion()
                log_dir = self.trainer.logger.log_dir
                ckpt_path = os.path.join(log_dir, "inversion.ckpt")
                print("\n==> Saving inversion states to", ckpt_path)
                torch.save(self.inversion.state_dict(), ckpt_path)
            else:
                print("\n==> Restoring inversion states from", ckpt_path)
                state = torch.load(ckpt_path, map_location=self.device)
                self.inversion.load_state_dict(state)
                self.hparams.inv_resume_from = None

    def training_step(self, batch, batch_idx):
        input, target = batch

        target_t = self.datamodule.transform_target(target)
        if self.model_old is not None:
            if self.inversion.training:
                self.inversion.eval()
            input_rh, target_rh = self.inversion.sample(input.shape[0])
            input = torch.cat([input, input_rh])
            target_t = torch.cat([target_t, target_rh])

        kwargs = dict(
            input=input,
            target=target_t,
            prediction=self(input),
        )

        if self.model_old is not None:
            if self.model_old.training:
                self.model_old.eval()

            feature = self.current_feature("pen")

            # finetuning classifier
            kwargs["input_ft"] = self.head.classify(feature.detach())
            kwargs["target_ft"] = target_t

            # weighted feature distillation
            kwargs["input_fkd"] = self.model_old.head.classify(feature)
            kwargs["target_fkd"] = self.model_old(input)

            # local classifier
            n_old, batch_size = self.model_old.head.num_classes, target.shape[0]
            kwargs["prediction"] = kwargs["prediction"][:batch_size, n_old:]
            kwargs["target"] = target_t[:batch_size] - n_old

        loss, loss_dict = self.compute_loss(**kwargs)

        self.log_dict({f"loss/{key}": val for key, val in loss_dict.items()})

        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.lr_factor,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
