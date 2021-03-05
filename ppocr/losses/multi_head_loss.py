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

import paddle
from paddle import nn


class MultiHeadLoss(nn.Layer):
    def __init__(self, reduction="mean", aux_head_loss_ratio_list=[], **kwargs):
        super(MultiHeadLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.reduction = reduction
        self.aux_head_loss_ratio_list = aux_head_loss_ratio_list
        assert reduction in ["mean", "sum", "none"
                             ], "reduction({}) format wrong!".format(reduction)

    def __call__(self, predicts, batch):
        loss_dict = dict()

        loss_dict["head_loss"] = self.calc_loss(predicts["head_out"], batch)
        for idx, aux_head in enumerate(predicts["aux_head_out"]):
            loss_dict["aux_head_loss_{}".format(idx)] = self.calc_loss(
                aux_head, batch) * self.aux_head_loss_ratio_list[idx]

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict

    def calc_loss(self, predicts, batch):
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')

        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
