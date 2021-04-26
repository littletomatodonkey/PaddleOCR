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


class ABF(nn.Layer):
    def __init__(self, s_ch, t_ch, out_ch):
        super(ABF, self).__init__()
        print("go into abf, s_ch: {}, t_ch: {}, out_ch: {}".format(s_ch, t_ch,
                                                                   out_ch))
        self.conv1 = nn.Conv2D(
            in_channels=s_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0)

        self.bn1 = nn.BatchNorm(num_channels=out_ch)

        self.conv2 = nn.Conv2D(
            in_channels=t_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0)

        self.bn2 = nn.BatchNorm(num_channels=out_ch)

        self.conv3 = nn.Conv2D(
            in_channels=2 * out_ch,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0)

        self.bn3 = nn.BatchNorm(num_channels=2)
        pass

    '''
    input:
        s_in: student input feat, low level
        t_in: teacher input feat, high level
    return
        t_out: teacher output feat
    '''

    def forward(self, s_in, t_in):
        h, w = s_in.shape[2:]
        s_in = self.conv1(s_in)
        s_in = self.bn1(s_in)

        t_in = self.conv2(t_in)
        t_in = F.adaptive_avg_pool2d(t_in, output_size=[h, w])
        t_in = self.bn2(t_in)

        ts_concat = paddle.concat([t_in, s_in], axis=1)

        ts_concat = self.conv3(ts_concat)
        ts_concat = self.bn3(ts_concat)

        s_in = paddle.multiply(s_in, ts_concat[:, :1, :, :])
        t_in = paddle.multiply(t_in, ts_concat[:, 1:, :, :])

        out = paddle.add(s_in, t_in)
        return out


class HCL(nn.Layer):
    def __init__(self, max_level=4, mode="max"):
        super(HCL, self).__init__()
        assert mode in ["max", "avg"]
        self.max_level = max_level
        self.mode = mode

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        h, w = x1.shape[2:]

        loss = F.mse_loss(x1, x2)
        for idx in range(1, self.max_level):
            target_h = max(h // pow(2, idx), 1)
            target_w = max(w // pow(2, idx), 1)
            if self.mode == "max":
                x1 = F.adaptive_max_pool2d(x1, output_size=[target_h, target_w])
                x2 = F.adaptive_max_pool2d(x2, output_size=[target_h, target_w])
            else:
                x1 = F.adaptive_avg_pool2d(x1, output_size=[target_h, target_w])
                x2 = F.adaptive_avg_pool2d(x2, output_size=[target_h, target_w])
            loss += F.mse_loss(x1, x2)
        return loss


class KnowLedgeReviewLoss(nn.Layer):
    '''
    A block in `Distilling Knowledge via Knowledge Review`
    See more in https://arxiv.org/abs/2104.09044    
    '''

    def __init__(self,
                 student_ch_num=[16, 24, 48, 288],
                 teacher_ch_num=[24, 40, 96, 576],
                 loss_ratio=1.0):
        '''
        channel_num:
            - small 0.5x : [16, 24, 48, 288]
            - small 1.0x : [24, 40, 96, 576]
            - small 1.25x: [32, 48, 120, 720]
        '''
        super(KnowLedgeReviewLoss, self).__init__()
        self.student_ch_num = student_ch_num[::-1]
        self.teacher_ch_num = teacher_ch_num[::-1]
        print("self.student_ch_num: {}, self.teacher_ch_num: {}".format(
            self.student_ch_num, self.teacher_ch_num))
        self.loss_ratio = loss_ratio

        self.hcl_loss_func = HCL(max_level=4, mode="max")

        self.conv = nn.Conv2D(
            in_channels=self.student_ch_num[0],
            out_channels=self.teacher_ch_num[0],
            kernel_size=1,
            stride=1,
            padding=0)

        self.abf_func_list = []
        for idx in range(1, len(self.student_ch_num)):
            self.abf_func_list.append(
                self.add_sublayer("abf_func_{}".format(idx),
                                  ABF(self.student_ch_num[
                                      idx], self.teacher_ch_num[idx - 1],
                                      self.teacher_ch_num[idx])))

    def forward(self, out1, out2):
        '''
        out1: student out dict
        out2: teacher out dict
        '''
        # high level --->>> low level
        s_backbone_list = list(out1.values())[::-1]
        t_backbone_list = list(out2.values())[::-1]
        assert len(s_backbone_list) == len(t_backbone_list) == len(
            self.teacher_ch_num) == len(self.student_ch_num)

        loss_dict = {}

        s_trans = self.conv(s_backbone_list[0])
        loss_dict["kr_loss_0"] = self.hcl_loss_func(
            s_trans, t_backbone_list[0]) * self.loss_ratio

        for idx in range(1, len(s_backbone_list)):
            s_trans = self.abf_func_list[idx - 1](s_backbone_list[idx], s_trans)
            loss_dict["kr_loss_{}".format(idx)] = self.hcl_loss_func(
                s_trans, t_backbone_list[idx]) * self.loss_ratio

        return loss_dict


def test_abf():
    bs = 32
    num_ch1 = 16
    num_ch2 = 16
    fm1_s = paddle.rand((bs, num_ch1, 16, 32))
    fm1_t = paddle.rand((bs, num_ch2, 32, 32))

    abf_func = ABF(num_ch1, num_ch2, 80)

    out = abf_func(fm1_t, fm1_s)
    print(out.shape)
    print("ok")
    return


def test_hcl():
    x1 = paddle.rand((32, 16, 16, 32))
    x2 = paddle.rand((32, 16, 16, 32))
    hcl_loss = HCL()
    loss = hcl_loss(x1, x2)
    print(loss)
    print("ok")
    return


def test_knowledge_review():
    s_channel = [16, 32, 64, 128]
    t_channel = [32, 64, 96, 256]
    s_out = {}
    t_out = {}
    for idx in range(4):
        s_out["st{}".format(idx)] = paddle.rand(
            [8, s_channel[idx], 64 // pow(2, idx), 64 // pow(2, idx)])
        t_out["st{}".format(idx)] = paddle.rand(
            [8, t_channel[idx], 64 // pow(2, idx), 64 // pow(2, idx)])

    kr_loss_func = KnowLedgeReviewLoss(
        student_ch_num=s_channel, teacher_ch_num=t_channel)

    loss_dict = kr_loss_func(s_out, t_out)
    print(loss_dict)

    print("ok")
    return


# if __name__ == "__main__":
# test_abf()
# test_hcl()
# test_knowledge_review()
