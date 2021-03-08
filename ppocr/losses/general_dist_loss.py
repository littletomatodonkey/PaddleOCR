# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
from paddle import nn
import paddle.nn.functional as F

from .rec_ctc_loss import CTCLoss
from .dml import DML
from .distillation_loss import InClassLoss

# more losses can refer to https://github.com/AberHu/Knowledge-Distillation-Zoo


class BaseLossClass(nn.Layer):
    def __init__(self, loss_type="l2loss", mode="mean", ratio=1.0):
        super(BaseLossClass, self).__init__()
        self.loss_type = loss_type
        self.mode = mode
        self.ratio = ratio

    def __call__(self, x, y):
        '''
            x: logits before softmax
            y: logits before softmax
        '''
        if self.loss_type == "l1loss":
            loss = F.l1_loss(x, y, mode=self.mode)
        elif self.loss_type == "l2loss":
            loss = F.mse_loss(x, y)
        elif self.loss_type == "cossim_loss":
            loss = F.cosine_similarity(x, y, axis=-1)
            loss = eval("paddle.{}(loss)".format(self.mode))
        elif self.loss_type == "celoss":
            y = F.softmax(y, axis=-1)
            loss = F.cross_entropy(x, y, soft_label=True)
            loss = eval("paddle.{}(loss)".format(self.mode))
        elif self.loss_type == "dmlloss":
            loss1 = self.dml_loss_func(x, y)
            loss2 = self.dml_loss_func(y, x)
            loss = (loss1 + loss2) / 2.0
        loss = loss * self.ratio
        return loss


class GeneralDistLoss(nn.Layer):
    def __init__(
            self,
            # distillation loss
            use_dist_loss=True,
            dist_loss_type="celoss",
            dist_loss_ratio=1.0,
            # student gt loss
            use_student_gt_loss=False,
            student_gt_loss_ratio=1.0,
            # set as true when tacher is not freezed (train from scratch)
            use_teacher_gt_loss=False,
            teacher_gt_loss_ratio=1.0,
            # inclass loss ratio
            use_inclass_loss=False,
            inclass_num_sections=4,
            inclass_loss_type="l2loss",
            inclass_loss_ratio=1.0,
            # whether to use backbone loss
            use_backbone_loss=False,
            backbone_loss_type="l2loss",
            backbone_loss_ratio=1.0,
            # other configs
    ):
        super(GeneralDistLoss, self).__init__()
        self.use_dist_loss = use_dist_loss
        self.use_student_gt_loss = use_student_gt_loss
        self.use_teacher_gt_loss = use_teacher_gt_loss
        self.use_inclass_loss = use_inclass_loss
        self.use_backbone_loss = use_backbone_loss

        if self.use_dist_loss:
            self.distillation_loss_func = BaseLossClass(
                loss_type=dist_loss_type, ratio=dist_loss_ratio)

        # used for either student or teacher gt
        self.ctc_loss_func = CTCLoss()

        if self.use_inclass_loss:
            self.inclass_loss_func = InClassLoss(
                num_sections=inclass_num_sections,
                loss_type=inclass_loss_type,
                loss_ratio=inclass_loss_ratio)

        if self.use_backbone_loss:
            self.backbone_loss_func = BaseLossClass(
                loss_type=backbone_loss_type, ratio=backbone_loss_ratio)

    def __call__(self, predicts, batch):
        teacher_list_out = predicts["teacher_list_out"]
        student_out = predicts["student_out"]

        loss_dict = dict()

        if self.use_dist_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["dist_loss_{}".format(
                    idx)] = self.distillation_loss_func(student_out["head_out"],
                                                        teacher_out["head_out"])

        if self.use_student_gt_loss:
            loss_dict["student_gt_loss"] = self.ctc_loss_func(
                student_out["head_out"], batch)["loss"]

        if self.use_teacher_gt_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["teacher_gt_loss_{}".format(
                    idx)] = self.ctc_loss_func(student_out["head_out"],
                                               batch)["loss"]

        if self.use_backbone_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["backbone_loss_{}".format(
                    idx)] = self.backbone_loss_func(student_out["backbone_out"],
                                                    teacher_out["backbone_out"])

        if self.use_inclass_loss:
            loss_dict["inclass_loss"] = self.inclass_loss_func(
                student_out["head_out"], batch)

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
