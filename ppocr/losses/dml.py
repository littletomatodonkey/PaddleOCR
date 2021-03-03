# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F


class DML(nn.Layer):
    '''
    Deep Mutual Learning
    https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
    '''

    def __init__(self, loss_ratio=1.0, **args):
        super(DML, self).__init__()
        self.loss_ratio = loss_ratio

    def forward(self, out1, out2):
        if isinstance(out1, dict):
            out1 = out1["head_out"]
        if isinstance(out2, dict):
            out2 = out2["head_out"]

        loss = F.kl_div(
            F.log_softmax(
                out1, axis=-1),
            F.softmax(
                out2, axis=-1),
            reduction='batchmean')
        loss = loss * self.loss_ratio
        return loss


if __name__ == "__main__":
    bs = 32
    num_ch1 = 64
    num_ch2 = 16
    fm1_s = paddle.rand((bs, num_ch1, 32, 32))
    fm1_t = paddle.rand((bs, num_ch1, 32, 32))

    dml_func = DML()
    loss = dml_func(fm1_s, fm1_t)
    print(loss)
