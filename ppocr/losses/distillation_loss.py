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


class InClassLoss(nn.Layer):
    def __init__(self):
        pass

    def __call__(self, x):
        pass


class DistillationLoss(nn.Layer):
    def __init__(self,
                 loss_type="celoss",
                 with_ctc_loss=False,
                 ctc_loss_ratio=0.5,
                 distillation_loss_ratio=0.5,
                 with_inclass_loss=False,
                 inclas_loss_type="l2loss",
                 inclas_loss_ratio=1.0,
                 blank_weight=None,
                 **kwargs):
        super(DistillationLoss, self).__init__()
        self.loss_type = loss_type
        self.with_ctc_loss = with_ctc_loss
        self.ctc_loss_ratio = ctc_loss_ratio
        self.distillation_loss_ratio = distillation_loss_ratio

        self.with_inclass_loss = with_inclass_loss
        self.inclas_loss_type = inclas_loss_type
        self.inclas_loss_ratio = inclas_loss_ratio

        self.blank_weight = blank_weight

        # TODO: add more loss
        supported_loss_type = ["celoss", "l2loss", "l1loss"]
        assert self.loss_type in supported_loss_type, "self.loss_type({}) must be in supported_loss_type({})".format(
            self.loss_type, supported_loss_type)

        supported_inclas_loss_type = ["l2loss"]
        assert self.inclas_loss_type in supported_loss_type, "self.inclas_loss_type({}) must be in supported_loss_type({})".format(
            self.inclas_loss_type, supported_inclas_loss_type)
        if self.with_ctc_loss:
            self.ctc_loss_func = CTCLoss()

    def __call__(self, predicts, batch):
        teacher_out = predicts["teacher_out"]
        student_out = predicts["student_out"]

        loss_weight = None
        if self.blank_weight is not None:
            loss_weight = paddle.ones((teacher_out.shape[-1], ))
            loss_weight[0] = self.blank_weight

        loss_dict = dict()
        if self.loss_type == "celoss":
            y = F.softmax(teacher_out, axis=-1)
            # with weighted loss
            if loss_weight is not None:
                loss_weight = loss_weight.reshape((1, 1, -1))
                y = paddle.multiply(y, loss_weight)
            cost = F.cross_entropy(student_out, y, soft_label=True)
            cost = paddle.mean(cost)
            loss_dict["celoss"] = cost * self.distillation_loss_ratio
        elif self.loss_type == "l2loss":
            cost = F.mse_loss(student_out, teacher_out)
            loss_dict["l2loss"] = cost * self.distillation_loss_ratio
        elif self.loss_type == "l1loss":
            cost = F.l1_loss(student_out, teacher_out, reduction='mean')
            loss_dict["l1loss"] = cost * self.distillation_loss_ratio
        else:
            assert False, "not supported loss type!"

        # data is split in 4 parts
        if self.with_inclass_loss:
            batch_size = teacher_out.shape[0] // 4
            teacher_batch_list = paddle.split(teacher_out, num_or_sections=4)
            student_batch_list = paddle.split(student_out, num_or_sections=4)

            if self.inclas_loss_type == "l2loss":
                loss_list = []
                for ii in range(len(student_batch_list)):
                    for jj in range(ii, len(student_batch_list)):
                        loss_list.append(
                            F.mse_loss(student_batch_list[ii],
                                       student_batch_list[jj]))

                cost = paddle.add_n(loss_list) / len(loss_list)
                loss_dict["inclass_l2loss"] = cost * self.inclas_loss_ratio
            else:
                assert False

        if self.with_ctc_loss:
            ctc_loss = self.ctc_loss_func(student_out,
                                          batch)["loss"] * self.ctc_loss_ratio
            loss_dict["ctcloss"] = ctc_loss

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


class SelfDistillationLoss(nn.Layer):
    def __init__(self,
                 num_section=4,
                 distillation_loss_type="l2loss",
                 distillation_loss_ratio=0.0,
                 with_ctc_loss=False,
                 ctc_loss_ratio=0.0,
                 **kwargs):
        super(SelfDistillationLoss, self).__init__()
        self.num_section = num_section
        self.with_ctc_loss = with_ctc_loss
        self.ctc_loss_ratio = ctc_loss_ratio
        self.distillation_loss_ratio = distillation_loss_ratio
        self.distillation_loss_type = distillation_loss_type

        supported_distillation_loss_type = ["l2loss"]
        assert distillation_loss_type in supported_distillation_loss_type, "distillation_loss_type({}) must be in supported_distillation_loss_type({})".format(
            self.distillation_loss_type, supported_distillation_loss_type)

        if with_ctc_loss:
            self.ctc_loss_func = CTCLoss(blank=0, reduction='none')

    def __call__(self, predicts, batch):
        loss_dict = {}

        # self distillation loss
        predicts_list = paddle.split(
            predicts, num_or_sections=self.num_section, axis=0)
        loss_list = []
        for ii in range(len(predicts_list)):
            for jj in range(ii, len(predicts_list)):
                if self.distillation_loss_type == "l2loss":
                    loss_list.append(
                        F.mse_loss(predicts_list[ii], predicts_list[jj]))
                else:
                    assert False
        cost = paddle.add_n(loss_list) / len(loss_list)
        loss_dict["inclass_l2loss"] = cost * self.distillation_loss_ratio

        # ctc loss
        if self.with_ctc_loss:
            ctc_loss = self.ctc_loss_func(predicts,
                                          batch)["loss"] * self.ctc_loss_ratio
            loss_dict["ctcloss"] = ctc_loss

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


class TDecodeCTCLoss(nn.Layer):
    def __init__(self, loss_ratio=1.0, **kwargs):
        super(TDecodeCTCLoss, self).__init__()
        self.loss_ratio = loss_ratio
        self.ctc_loss_func = CTCLoss(blank=0, reduction='none')

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
                    conf_list.append(1)
            result_char_list.append(char_list)
            result_conf_list.append(np.mean(conf_list))
        return result_char_list, result_conf_list

    def __call__(self, predicts, batch):
        '''
        the gt is got from predicts
        '''
        teacher_out = predicts["teacher_out"]
        student_out = predicts["student_out"]

        loss_dict = {}

        y = F.softmax(teacher_out, axis=-1)
        preds_label = y.argmax(axis=2).numpy()
        preds_prob = y.max(axis=2).numpy()

        max_length = batch[1].shape[1]

        decoded_preds_label, preds_prob = self.decode_teacher(
            preds_label, preds_prob, is_remove_duplicate=False)
        predicts_len_list = [
            min(max_length, len(pred)) for pred in decoded_preds_label
        ]
        predicts_len_list = paddle.to_tensor(predicts_len_list, dtype="int64")
        for idx in range(len(decoded_preds_label)):
            decoded_preds_label[idx].extend([0] * (
                max_length - len(decoded_preds_label[idx])))
            decoded_preds_label[idx] = decoded_preds_label[idx][:max_length]

        decoded_preds_label = paddle.to_tensor(
            decoded_preds_label, dtype="int64")
        preds_prob = np.array(preds_prob).astype("float32")

        teacher_batch = batch
        teacher_batch[1] = decoded_preds_label
        teacher_batch[2] = predicts_len_list
        loss_dict["decode_ctc_loss"] = self.ctc_loss_func(
            student_out, teacher_batch)["loss"] * self.loss_ratio

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
