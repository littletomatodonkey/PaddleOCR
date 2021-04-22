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
from .distillation_loss import jsdiv_me
from .fsp import FSP


def dml_me_loss(out1, out2):
    '''
    for backbone or head out
    backbone: bs x ch(288) x 1 x ts(80)
    head    : bs x ts(80) x num_classes
    '''
    # if isinstance(out1, dict):
    #     out1 = out1["head_out"]
    # if isinstance(out2, dict):
    #     out2 = out2["head_out"]
    soft_out1 = F.softmax(out1, axis=-1)
    log_soft_out1 = paddle.log(soft_out1)

    soft_out2 = F.softmax(out2, axis=-1)
    log_soft_out2 = paddle.log(soft_out2)

    loss = (F.kl_div(
        log_soft_out1, soft_out2, reduction='batchmean') + F.kl_div(
            log_soft_out2, soft_out1, reduction='batchmean')) / 2.0

    return loss


def dml_sigmoid_loss(out1, out2):
    '''
    for backbone or head out
    backbone: bs x ch(288) x 1 x ts(80)
    head    : bs x ts(80) x num_classes
    consider of multiple crests
    '''
    soft_out1 = F.sigmoid(out1)
    log_soft_out1 = paddle.log(soft_out1)

    soft_out2 = F.sigmoid(out2)
    log_soft_out2 = paddle.log(soft_out2)

    loss = (F.kl_div(
        log_soft_out1, soft_out2, reduction='batchmean') + F.kl_div(
            log_soft_out2, soft_out1, reduction='batchmean')) / 2.0

    return loss


class BaseLossClass(nn.Layer):
    def __init__(self, loss_type="l2loss", mode="mean", ratio=1.0):
        super(BaseLossClass, self).__init__()
        self.loss_type = loss_type
        self.mode = mode
        self.ratio = ratio
        if self.loss_type == "dmlloss":
            self.dml_loss_func = DML(1.0)

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
        elif self.loss_type == "jsdiv_me_loss":
            loss = jsdiv_me(x, y)
        elif self.loss_type == "dml_me_loss":
            loss = dml_me_loss(x, y)
        elif self.loss_type == "dml_sigmoid_loss":
            loss = dml_sigmoid_loss(x, y)
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
            # if use_partial_data_for_other_loss is set as true,
            # just one part of data will be use to calc loss except inclass loss
            use_partial_data_for_other_loss=False,
            inclass_num_sections=4,
            inclass_loss_type="l2loss",
            inclass_loss_ratio=1.0,
            # whether to use backbone loss
            use_backbone_loss=False,
            backbone_loss_type="l2loss",
            backbone_loss_ratio=1.0,

            # whether to use neck loss
            use_neck_loss=False,
            neck_loss_type="l2loss",
            neck_loss_ratio=1.0,

            # whether to use fsp in backbone and neack out
            use_fsp_loss_backbone_neck=False,
            fsp_loss_backbone_neck_ratio=1.0,

            # whether to use teacher loss
            # if freeze_teacher is False, we should calc teacher loss between themselves
            freeze_teacher=True,

            # if is true, model merge will be used for supervision
            # recommanded in DML
            use_model_merge_dist_loss=False,

            # with unlabeled data (gt label is ###)
            use_unlabeled_data=False,
            # data normalize is used when the unlabeled data contains
            normalize_ctc_loss=False,

            # other configs
    ):
        super(GeneralDistLoss, self).__init__()
        self.use_dist_loss = use_dist_loss
        self.use_student_gt_loss = use_student_gt_loss
        self.use_teacher_gt_loss = use_teacher_gt_loss
        self.use_inclass_loss = use_inclass_loss
        self.use_neck_loss = use_neck_loss
        self.use_backbone_loss = use_backbone_loss
        self.freeze_teacher = freeze_teacher
        self.use_model_merge_dist_loss = use_model_merge_dist_loss
        self.use_unlabeled_data = use_unlabeled_data
        self.normalize_ctc_loss = normalize_ctc_loss
        self.use_fsp_loss_backbone_neck = use_fsp_loss_backbone_neck
        self.use_partial_data_for_other_loss = use_partial_data_for_other_loss
        self.inclass_num_sections = inclass_num_sections

        if self.use_dist_loss:
            self.distillation_loss_func = BaseLossClass(
                loss_type=dist_loss_type, ratio=dist_loss_ratio)

        # used for either student or teacher gt
        self.ctc_loss_func = CTCLoss()
        self.ctc_loss_func_raw = CTCLoss(reduction="none")

        if self.use_inclass_loss:
            self.inclass_loss_func = InClassLoss(
                num_sections=inclass_num_sections,
                loss_type=inclass_loss_type,
                loss_ratio=inclass_loss_ratio)

        if self.use_backbone_loss:
            self.backbone_loss_func = BaseLossClass(
                loss_type=backbone_loss_type, ratio=backbone_loss_ratio)

        if self.use_neck_loss:
            self.neck_loss_func = BaseLossClass(
                loss_type=neck_loss_type, ratio=neck_loss_ratio)

        if self.use_fsp_loss_backbone_neck:
            self.fsp_loss_backbone_neck_func = FSP(
                loss_ratio=fsp_loss_backbone_neck_ratio)

    def calc_ignore_flag(self, batch):
        '''
            batch[0]: predicts, bs x 25 x 6625
            batch[1]: gt_label, bs x 25
            batch[2]: gt_len,   bs x 1
        '''
        gt_label = batch[1].numpy()
        ignore_arr = np.zeros_like(batch[1].numpy())
        # fake label is ###
        ignore_arr[:, :3] = 5461
        ignore_flag = np.logical_not(
            np.alltrue(
                ignore_arr == gt_label, axis=1)).astype("float32")
        return ignore_flag

    def calc_ctc_loss(self, predict, batch):
        if self.use_unlabeled_data:
            ctc_loss = self.ctc_loss_func_raw(predict["head_out"],
                                              batch)["loss"]
            ignore_flag = self.calc_ignore_flag(batch)
            ctc_loss = ctc_loss * paddle.to_tensor(ignore_flag)
            ctc_loss = ctc_loss.sum()
            # actuall batch that the ctc loss can be calc
            valid_bs = np.sum(ignore_flag)
            if self.normalize_ctc_loss and valid_bs != 0:
                ctc_loss = ctc_loss / valid_bs
            else:
                ctc_loss = ctc_loss / ctc_loss.shape[0]

        else:
            ctc_loss = self.ctc_loss_func(predict["head_out"], batch)["loss"]
        return ctc_loss

    def __call__(self, ori_predicts, ori_batch):
        # re-generate predicts
        if self.use_partial_data_for_other_loss:
            actual_bs = ori_batch[0].shape[0] // self.inclass_num_sections
            # predicts
            predicts = {"teacher_list_out": [], "student_out": None}
            for ori_t_map in ori_predicts["teacher_list_out"]:
                predicts["teacher_list_out"].append(
                    {key: ori_t_map[key][:actual_bs]
                     for key in ori_t_map})
            predicts["student_out"] = {
                key: ori_predicts["student_out"][:actual_bs]
                for key in ori_predicts["student_out"]
            }
            batch = [gt[:actual_bs] for gt in ori_batch]
        else:
            predicts = ori_predicts
            batch = ori_batch

        teacher_list_out = predicts["teacher_list_out"]
        student_out = predicts["student_out"]

        # for key in student_out:
        #     print("{}, shape: {}".format(key, student_out[key].shape))
        # exit()

        loss_dict = dict()

        if self.use_dist_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["dist_loss_{}".format(
                    idx)] = self.distillation_loss_func(student_out["head_out"],
                                                        teacher_out["head_out"])

            # commonly used in DML loss, cause they are learned from scratch
            if not self.freeze_teacher:
                for row in range(len(teacher_list_out)):
                    for col in range(row + 1, len(teacher_list_out)):
                        loss_dict["dist_in_teacher_loss_{}_{}".format(
                            row, col)] = self.distillation_loss_func(
                                teacher_list_out[row]["head_out"],
                                teacher_list_out[col]["head_out"])

            if self.use_model_merge_dist_loss:
                res_list = [
                    teacher_out["head_out"] for teacher_out in teacher_list_out
                ] + [student_out["head_out"]]
                merge_result = paddle.add_n(res_list) / len(res_list)
                loss_dict["model_merge_dist_loss_{}".format(
                    idx)] = self.distillation_loss_func(student_out["head_out"],
                                                        merge_result)

        if self.use_student_gt_loss:
            loss_dict["student_gt_loss"] = self.calc_ctc_loss(student_out,
                                                              batch)

        if self.use_teacher_gt_loss:
            for idx, teacher_out, in enumerate(teacher_list_out):
                loss_dict["teacher_gt_loss_{}".format(
                    idx)] = self.calc_ctc_loss(teacher_out, batch)

        if self.use_backbone_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["backbone_loss_{}".format(
                    idx)] = self.backbone_loss_func(student_out["backbone_out"],
                                                    teacher_out["backbone_out"])

            # commonly used in DML loss, cause they are learned from scratch
            if not self.freeze_teacher:
                for row in range(len(teacher_list_out)):
                    for col in range(row + 1, len(teacher_list_out)):
                        loss_dict["backbone_in_teacher_loss_{}_{}".format(
                            row, col)] = self.backbone_loss_func(
                                teacher_list_out[row]["backbone_out"],
                                teacher_list_out[col]["backbone_out"])

        if self.use_neck_loss:
            for idx, teacher_out in enumerate(teacher_list_out):
                loss_dict["neck_loss_{}".format(idx)] = self.backbone_loss_func(
                    student_out["neck_out"], teacher_out["neck_out"])
            # commonly used in DML loss, cause they are learned from scratch
            if not self.freeze_teacher:
                for row in range(len(teacher_list_out)):
                    for col in range(row + 1, len(teacher_list_out)):
                        loss_dict["neck_in_teacher_loss_{}_{}".format(
                            row, col)] = self.backbone_loss_func(
                                teacher_list_out[row]["neck_out"],
                                teacher_list_out[col]["neck_out"])

        # for inclass loss, loss should be computed for the total batch
        if self.use_inclass_loss:
            loss_dict["student_inclass_loss"] = self.inclass_loss_func(
                ori_predicts["student_out"]["head_out"], ori_batch)

            if not self.freeze_teacher:
                for idx, teacher_out, in enumerate(ori_predicts[
                        "teacher_list_out"]):
                    loss_dict["teacher_inclass_loss_{}".format(
                        idx)] = self.inclass_loss_func(teacher_out["head_out"],
                                                       ori_batch)

        if self.use_fsp_loss_backbone_neck:
            for idx, teacher_out, in enumerate(teacher_list_out):
                # 128 x 288 x 1 x 80
                fm_t1 = teacher_out["backbone_out"]
                # 128 x 80 x 96
                fm_t2 = teacher_out["neck_out"]

                fm_s1 = student_out["backbone_out"]
                fm_s2 = student_out["neck_out"]

                # 128 x 80 x 96 -> 128 x 96 x 80 -> 128 x 96 x 1 x 80
                fm_t2 = fm_t2.transpose([0, 2, 1]).unsqueeze(axis=2)
                fm_s2 = fm_s2.transpose([0, 2, 1]).unsqueeze(axis=2)

                loss_dict["fsp_backbone_neck_loss_{}".format(
                    idx)] = self.fsp_loss_backbone_neck_func(fm_s1, fm_s2,
                                                             fm_t1, fm_t2)

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
