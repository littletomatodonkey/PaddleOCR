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


class FSP(nn.Layer):
    '''
    A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
    '''

    def __init__(self, loss_ratio=1.0, **args):
        super(FSP, self).__init__()
        self.loss_ratio = loss_ratio

    def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
        loss = F.mse_loss(
            self.fsp_matrix(fm_s1, fm_s2), self.fsp_matrix(fm_t1, fm_t2))
        loss = self.loss_ratio * loss
        return loss

    def fsp_matrix(self, fm1, fm2):
        if fm1.shape[2] != fm2.shape[2]:  # improve
            # if fm1.shape[2] > fm2.shape[2]: original code
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.shape[2], fm2.shape[3]))
        fm1 = fm1.reshape([fm1.shape[0], fm1.shape[1], -1])
        fm2 = fm2.reshape([fm2.shape[0], fm2.shape[1], -1]).transpose([0, 2, 1])
        fsp = paddle.bmm(fm1, fm2) / fm1.shape[2]  # N x C1 x C2

        return fsp


if __name__ == "__main__":
    bs = 32
    num_ch1 = 64
    num_ch2 = 16
    fm1_s = paddle.rand((bs, num_ch1, 16, 16))
    fm2_s = paddle.rand((bs, num_ch2, 32, 32))

    fm1_t = paddle.rand((bs, num_ch1, 32, 32))
    fm2_t = paddle.rand((bs, num_ch2, 64, 32))

    fsp_func = FSP()
    loss = fsp_func(fm1_s, fm2_s, fm1_t, fm2_t)
    print(loss)
