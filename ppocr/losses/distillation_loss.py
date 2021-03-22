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

# more losses can refer to https://github.com/AberHu/Knowledge-Distillation-Zoo


def _kldiv(input, target, eps=1.0e-10):
    cost = target * paddle.log((target + eps) / (input + eps)) * input.shape[-1]
    return cost


# refer to PaddleClas
def jsdiv_me(input, target):
    input = F.softmax(input)
    target = F.softmax(target)
    cost = _kldiv(input, target) + _kldiv(target, input)
    cost = cost / 2
    avg_cost = paddle.mean(cost)
    return avg_cost


def balanced_l1_loss(pred,
                     target,
                     beta=0.3,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    # beta=0.3, is got when we find the initial value is about 0.34
    assert beta > 0
    diff = paddle.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = paddle.where(
        diff < beta, alpha / b *
        (b * diff + 1) * paddle.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)
    loss = loss.mean()
    return loss


class InClassLoss(nn.Layer):
    def __init__(self,
                 num_sections=4,
                 loss_ratio=1.0,
                 loss_type="l2loss",
                 **args):
        self.num_sections = num_sections
        self.loss_ratio = loss_ratio
        self.loss_type = loss_type

        supported_types = [
            "l1loss", "l2loss", "cossim_loss", "jsdiv_me_loss", "libra_loss"
        ]
        assert loss_type in supported_types, "loss type must be in {} but got {}".format(
            supported_types, loss_type)

    def __call__(self, predicts, batch):
        # self distillation loss
        predicts_list = paddle.split(
            predicts, num_or_sections=self.num_sections, axis=0)
        loss_list = []
        for ii in range(len(predicts_list)):
            for jj in range(ii, len(predicts_list)):
                if self.loss_type == "l1loss":
                    loss_list.append(
                        F.l1_loss(predicts_list[ii], predicts_list[jj]))
                elif self.loss_type == "l2loss":
                    loss_list.append(
                        F.mse_loss(predicts_list[ii], predicts_list[jj]))
                elif self.loss_type == "cossim_loss":
                    curr_loss = F.cosine_similarity(
                        predicts_list[ii], predicts_list[jj], axis=-1)
                    loss_list.append(curr_loss.mean())
                elif self.loss_type == "jsdiv_me_loss":
                    curr_loss = jsdiv_me(predicts_list[ii], predicts_list[jj])
                    loss_list.append(curr_loss.mean())
                elif self.loss_type == "libra_loss":
                    curr_loss = balanced_l1_loss(predicts_list[ii],
                                                 predicts_list[jj])
                    loss_list.append(curr_loss.mean())
                else:
                    assert False
        cost = paddle.add_n(loss_list) / len(loss_list)
        return cost * self.loss_ratio


class DistillationLoss(nn.Layer):
    def __init__(
            self,
            loss_type="celoss",
            with_student_ctc_loss=False,
            ctc_loss_ratio=0.5,
            distillation_loss_ratio=0.5,
            with_inclass_loss=False,
            inclass_loss_type="l2loss",
            inclass_loss_ratio=1.0,
            inclass_num_sections=4,
            blank_weight=None,
            with_teacher_ctc_loss=False,
            dml_loss_ratio=0.5,
            # if set as true, (t+s)/2 will be used for dml
            use_combined_ts_loss=False,
            combined_ts_loss_ratio=0.5,
            with_backbone_loss=False,
            backbone_loss_type="l2loss",
            backbone_loss_ratio=0.5,
            **kwargs):
        super(DistillationLoss, self).__init__()
        self.loss_type = loss_type
        self.with_student_ctc_loss = with_student_ctc_loss
        self.ctc_loss_ratio = ctc_loss_ratio
        self.distillation_loss_ratio = distillation_loss_ratio

        self.with_inclass_loss = with_inclass_loss
        self.inclass_loss_type = inclass_loss_type
        self.inclass_loss_ratio = inclass_loss_ratio
        self.inclass_num_sections = inclass_num_sections

        self.blank_weight = blank_weight

        # for DML loss
        self.with_teacher_ctc_loss = with_teacher_ctc_loss
        self.use_dml_loss = self.loss_type == "dmlloss"
        self.dml_loss_ratio = dml_loss_ratio
        # when using dml loss, combined sup performs better
        self.use_combined_ts_loss = use_combined_ts_loss
        self.combined_ts_loss_ratio = combined_ts_loss_ratio

        if self.use_dml_loss:
            self.dml_loss_func = DML(self.dml_loss_ratio)

        self.with_backbone_loss = with_backbone_loss
        self.backbone_loss_type = backbone_loss_type
        self.backbone_loss_ratio = backbone_loss_ratio

        # TODO: add more loss
        supported_loss_type = ["celoss", "l2loss", "l1loss", "dmlloss"]
        assert self.loss_type in supported_loss_type, "self.loss_type({}) must be in supported_loss_type({})".format(
            self.loss_type, supported_loss_type)

        # build inclass loss
        if self.with_inclass_loss:
            self.inclass_loss_func = InClassLoss(
                num_sections=self.inclass_num_sections,
                loss_ratio=self.inclass_loss_ratio,
                loss_type=self.inclass_loss_type)

        if self.with_student_ctc_loss:
            self.ctc_loss_func = CTCLoss()

        # build backbone loss to supervise the 
        if self.with_backbone_loss:
            assert self.backbone_loss_type in ["l1loss", "l2loss"]
            if self.backbone_loss_type == "l1loss":
                self.backbone_loss_func = nn.L1Loss(reduction="mean")
            elif self.backbone_loss_type == "l2loss":
                self.backbone_loss_func = nn.MSELoss(reduction="mean")
            else:
                print("not supported backbone_loss_type: {}".format(
                    self.backbone_loss_type))
                exit()

    def __call__(self, predicts, batch):
        teacher_out = predicts["teacher_out"]
        student_out = predicts["student_out"]

        loss_weight = None
        if self.blank_weight is not None:
            loss_weight = paddle.ones((teacher_out.shape[-1], ))
            loss_weight[0] = self.blank_weight

        loss_dict = dict()
        if self.loss_type == "celoss":
            y = F.softmax(teacher_out["head_out"], axis=-1)
            # with weighted loss
            if loss_weight is not None:
                loss_weight = loss_weight.reshape((1, 1, -1))
                y = paddle.multiply(y, loss_weight)
            cost = F.cross_entropy(student_out["head_out"], y, soft_label=True)
            cost = paddle.mean(cost)
            loss_dict["celoss"] = cost * self.distillation_loss_ratio
        elif self.loss_type == "l2loss":
            cost = F.mse_loss(student_out["head_out"], teacher_out["head_out"])
            loss_dict["l2loss"] = cost * self.distillation_loss_ratio
        elif self.loss_type == "l1loss":
            cost = F.l1_loss(
                student_out["head_out"],
                teacher_out["head_out"],
                reduction='mean')
            loss_dict["l1loss"] = cost * self.distillation_loss_ratio
        elif self.loss_type == "dmlloss":
            cost1 = self.dml_loss_func(student_out, teacher_out)
            cost2 = self.dml_loss_func(teacher_out, student_out)
            cost = (cost1 + cost2) / 2.0
            loss_dict["dmlloss"] = cost

            if self.use_combined_ts_sup:
                combined_feat = (
                    student_out["head_out"] + teacher_out["head_out"]) / 2
                cost3 = self.dml_loss_func(student_out, combined_feat)
                cost4 = self.dml_loss_func(combined_feat, student_out)
                cost5 = self.dml_loss_func(teacher_out, combined_feat)
                cost6 = self.dml_loss_func(combined_feat, teacher_out)
                # avg if half of the mormal dml loss
                cost = cost + (cost3 + cost4 + cost5 + cost6
                               ) / 4.0 * self.combined_ts_loss_ratio

        else:
            assert False, "not supported loss type!"

        if self.with_inclass_loss:
            cost = self.inclass_loss_func(student_out["head_out"], batch)
            loss_dict["inclass_{}".format(self.inclass_loss_type)] = cost

        if self.with_student_ctc_loss:
            student_ctc_loss = self.ctc_loss_func(
                student_out, batch)["loss"] * self.ctc_loss_ratio
            loss_dict["student_ctcloss"] = student_ctc_loss

        if self.with_teacher_ctc_loss:
            teacher_ctc_loss = self.ctc_loss_func(
                teacher_out, batch)["loss"] * self.ctc_loss_ratio
            loss_dict["teacher_ctcloss"] = teacher_ctc_loss

        if self.with_backbone_loss:
            teacher_backbone = predicts["teacher_out"]["backbone_out"]
            student_backbone = predicts["student_out"]["backbone_out"]
            loss_dict["backbone_loss"] = self.backbone_loss_func(
                student_backbone, teacher_backbone) * self.backbone_loss_ratio

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


class SelfDistillationLoss(nn.Layer):
    def __init__(self,
                 num_sections=4,
                 distillation_loss_type="l2loss",
                 distillation_loss_ratio=1.0,
                 with_ctc_loss=False,
                 ctc_loss_ratio=1.0,
                 ctc_loss_mode="mean",
                 loss_after_softmax=False,
                 **kwargs):
        super(SelfDistillationLoss, self).__init__()
        self.num_sections = num_sections
        self.with_ctc_loss = with_ctc_loss
        self.ctc_loss_ratio = ctc_loss_ratio
        self.distillation_loss_ratio = distillation_loss_ratio
        self.distillation_loss_type = distillation_loss_type
        self.loss_after_softmax = loss_after_softmax
        self.ctc_loss_mode = ctc_loss_mode

        if with_ctc_loss:
            self.ctc_loss_func = CTCLoss(blank=0, reduction=self.ctc_loss_mode)

        self.in_class_loss_func = InClassLoss(
            num_sections=num_sections,
            loss_ratio=distillation_loss_ratio,
            loss_type=distillation_loss_type)

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

    def __call__(self, predicts, batch):
        loss_dict = {}

        if self.loss_after_softmax:
            out = F.softmax(predicts, axis=-1)
        else:
            out = predicts
        loss_dict["inclass_{}".format(
            self.distillation_loss_type)] = self.in_class_loss_func(out, batch)

        # ctc loss
        if self.with_ctc_loss:
            # 
            ctc_loss = self.ctc_loss_func(predicts, batch)["loss"]
            ignore_flag = self.calc_ignore_flag(batch)
            ctc_loss = ctc_loss * paddle.to_tensor(ignore_flag)
            ctc_loss = ctc_loss.mean()
            loss_dict["ctcloss"] = ctc_loss * self.ctc_loss_ratio

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


class TDecodeCTCLoss(nn.Layer):
    def __init__(self,
                 loss_ratio=1.0,
                 use_weight_loss=False,
                 weight_loss_type="direct",
                 use_inclass_loss=False,
                 inclass_loss_ratio=1.0,
                 inclass_loss_type="l2loss",
                 **kwargs):
        super(TDecodeCTCLoss, self).__init__()
        self.loss_ratio = loss_ratio
        self.use_weight_loss = use_weight_loss
        self.weight_loss_type = weight_loss_type
        self.ctc_loss_func = CTCLoss(blank=0, reduction='none')
        self.use_inclass_loss = use_inclass_loss
        self.inclass_loss_ratio = inclass_loss_ratio
        self.inclass_loss_type = inclass_loss_type

        # direct: loss * weight
        # exp_minux: loss * exp(1.0-weight)
        assert weight_loss_type in ["direct", "exp_minus"]

        if self.use_inclass_loss:
            self.inclass_loss_func = InClassLoss(
                num_sections=4,
                loss_ratio=self.inclass_loss_ratio,
                loss_type=self.inclass_loss_type)

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    # refer to ppocr/postprocess/rec_postprocess.py
    def decode_teacher(self,
                       text_index,
                       text_prob=None,
                       is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_char_list = []
        result_conf_list = []

        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(text_index[batch_idx][idx])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1.0)
            result_char_list.append(char_list)
            result_conf_list.append(np.mean(conf_list))
        return result_char_list, result_conf_list

    def __call__(self, predicts, batch):
        '''
        the gt is got from predicts
        '''
        teacher_out = predicts["teacher_out"]["head_out"]
        student_out = predicts["student_out"]["head_out"]

        loss_dict = {}

        y = F.softmax(teacher_out, axis=-1)
        preds_label = y.argmax(axis=2).numpy()
        preds_prob = y.max(axis=2).numpy()

        max_length = batch[1].shape[1]

        decoded_preds_label, preds_prob = self.decode_teacher(
            preds_label, preds_prob, is_remove_duplicate=True)
        predicts_len_list = [
            min(max_length, len(pred)) for pred in decoded_preds_label
        ]
        # preds_prob = [1.0] * len(preds_prob)

        predicts_len_list = paddle.to_tensor(predicts_len_list, dtype="int64")
        for idx in range(len(decoded_preds_label)):
            decoded_preds_label[idx].extend([0] * (
                max_length - len(decoded_preds_label[idx])))
            decoded_preds_label[idx] = decoded_preds_label[idx][:max_length]

        decoded_preds_label = paddle.to_tensor(
            decoded_preds_label, dtype="int64")

        preds_prob = paddle.to_tensor(preds_prob, dtype="float32")
        # prevent model from multiply nan
        preds_prob = paddle.where(
            paddle.isnan(preds_prob), paddle.ones_like(preds_prob), preds_prob)

        teacher_batch = batch
        #         teacher_batch[1] = decoded_preds_label
        #         teacher_batch[2] = predicts_len_list

        loss = self.ctc_loss_func(student_out, teacher_batch)["loss"]

        if self.use_weight_loss:
            if self.weight_loss_type == "exp_minus":
                preds_prob = paddle.clip(preds_prob, min=0.6, max=1.0)
                preds_prob = paddle.exp(1 - preds_prob)
            elif self.weight_loss_type == "direct":
                preds_prob = preds_prob
            loss = paddle.multiply(loss, preds_prob)
            # print("use_weight: ", loss)

        if self.use_inclass_loss:
            inclass_loss = self.inclass_loss_func(student_out, batch)
            loss_dict["inclass_loss"] = inclass_loss

        loss_dict["decode_ctc_loss"] = loss.mean() * self.loss_ratio

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


class AllFeatLoss(nn.Layer):
    def __init__(self, **args):
        super(AllFeatLoss, self).__init__()

    def __call__(self, predicts, batch):
        '''
        the gt is got from predicts
        '''
        teacher_out = predicts["teacher_out"]["head_out"]
        student_out = predicts["student_out"]["head_out"]

        loss_dict = {}

        y = F.softmax(teacher_out, axis=-1)
        preds_label = y.argmax(axis=2).numpy()
        preds_prob = y.max(axis=2).numpy()

        max_length = batch[1].shape[1]

        decoded_preds_label, preds_prob = self.decode_teacher(
            preds_label, preds_prob, is_remove_duplicate=True)
        predicts_len_list = [
            min(max_length, len(pred)) for pred in decoded_preds_label
        ]
        # preds_prob = [1.0] * len(preds_prob)

        predicts_len_list = paddle.to_tensor(predicts_len_list, dtype="int64")
        for idx in range(len(decoded_preds_label)):
            decoded_preds_label[idx].extend([0] * (
                max_length - len(decoded_preds_label[idx])))
            decoded_preds_label[idx] = decoded_preds_label[idx][:max_length]

        decoded_preds_label = paddle.to_tensor(
            decoded_preds_label, dtype="int64")

        preds_prob = paddle.to_tensor(preds_prob, dtype="float32")
        # prevent model from multiply nan
        preds_prob = paddle.where(
            paddle.isnan(preds_prob), paddle.ones_like(preds_prob), preds_prob)

        teacher_batch = batch
        #         teacher_batch[1] = decoded_preds_label
        #         teacher_batch[2] = predicts_len_list

        loss = self.ctc_loss_func(student_out, teacher_batch)["loss"]

        if self.use_weight_loss:
            if self.weight_loss_type == "exp_minus":
                preds_prob = paddle.clip(preds_prob, min=0.6, max=1.0)
                preds_prob = paddle.exp(1 - preds_prob)
            elif self.weight_loss_type == "direct":
                preds_prob = preds_prob
            loss = paddle.multiply(loss, preds_prob)
            # print("use_weight: ", loss)

        loss_dict["decode_ctc_loss"] = loss.mean() * self.loss_ratio

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
