# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppocr.modeling.backbones.det_mobilenet_v3 import make_divisible
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer as ConvBN

__all__ = ['RepMobileNetV3']


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        if paddle.__version__ == "2.0.0-rc1":
            outputs = F.activation.hard_sigmoid(outputs)
        else:
            outputs = F.activation.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


class RepVGGBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 name=None):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.kernel_size = kernel_size

        assert kernel_size in [3, 5]
        assert padding == kernel_size // 2

        padding_11 = padding - kernel_size // 2

        self.rbr_identity = nn.BatchNorm2D(
            num_features=in_channels
        ) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            if_act=False,
            name=name + "_rbr_dense")
        self.rbr_1x1 = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups,
            if_act=False,
            name=name + "rbr_1x1")

    def forward(self, inputs):
        if not self.training:
            return self.rbr_reparam(inputs)

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out

    def eval(self):
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                padding_mode=self.padding_mode)
        self.training = False
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        for layer in self.sublayers():
            layer.eval()

    def get_equivalent_kernel_bias(self):
        kernel_kxk, bias_kxk = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel_kxk + self._pad_1x1_to_kxk_tensor(
            kernel_1x1) + kernelid, bias_kxk + bias_1x1 + biasid

    def _pad_1x1_to_kxk_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            p = (self.kernel_size - 1) // 2
            return F.pad(kernel_1x1, [p, p, p, p])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, self.kernel_size,
                     self.kernel_size),
                    dtype=np.float32)
                center_idx = (self.kernel_size - 1) // 2
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, center_idx, center_idx] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        print("[gry debug]kernel size: {}, padding: {}".format(kernel_size,
                                                               padding))
        if kernel_size in [3, 5]:
            print("trying to use rep vgg block....")
            self.conv = RepVGGBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                name=name + "_repblock")
            self.bn = None
        else:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                weight_attr=ParamAttr(name=name + '_weights'),
                bias_attr=False)
            self.bn = nn.BatchNorm(
                num_channels=out_channels,
                act=None,
                param_attr=ParamAttr(name=name + "_bn_scale"),
                bias_attr=ParamAttr(name=name + "_bn_offset"),
                moving_mean_name=name + "_bn_mean",
                moving_variance_name=name + "_bn_variance")

        if self.if_act and self.act == "meta_acon":
            self.meta_acon_func = MetaAconC(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hard_swish":
                if paddle.__version__ == "2.0.0-rc1":
                    x = F.activation.hard_swish(x)
                else:
                    x = F.activation.hardswish(x)
            elif self.act == "meta_acon":
                x = self.meta_acon_func(x)
            else:
                print("The activation function is selected incorrectly.")
                # print("act: ", self.act)
                exit()
        return x

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()


class RepMobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 last_act="hard_swish",
                 prefix_name="",
                 embedding_size=None,
                 pool_kernel_size=2,
                 return_feat_dict=False,
                 add_pre_act_layer=False,
                 **kwargs):
        super().__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        elif model_name == "small_deeper":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [3, 88, 24, False, 'relu', 1],  # added deeper
                [3, 88, 24, False, 'relu', 1],  # added deeper
                [3, 88, 24, False, 'relu', 1],  # added deeper
                [5, 96, 40, True, 'hard_swish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],  # added deeper
                [5, 144, 48, True, 'hard_swish', 1],  # added deeper
                [5, 144, 48, True, 'hard_swish', 1],  # added deeper
                [5, 288, 96, True, 'hard_swish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],  # added deeper
                [5, 576, 96, True, 'hard_swish', 1],  # added deeper
            ]
            cls_ch_squeeze = 576
        elif model_name == "small_acon":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'meta_acon', (small_stride[2], 1)],
                [5, 240, 40, True, 'meta_acon', 1],
                [5, 240, 40, True, 'meta_acon', 1],
                [5, 120, 48, True, 'meta_acon', 1],
                [5, 144, 48, True, 'meta_acon', 1],
                [5, 288, 96, True, 'meta_acon', (small_stride[3], 1)],
                [5, 576, 96, True, 'meta_acon', 1],
                [5, 576, 96, True, 'meta_acon', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        self.cfg = cfg
        self.return_feat_dict = return_feat_dict
        self.add_pre_act_layer = add_pre_act_layer

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hard_swish',
            name=prefix_name + 'conv1')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name=prefix_name + 'conv' + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.out_channels = make_divisible(scale * cls_ch_squeeze)
        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=False,
            name=prefix_name + 'conv_last')

        self.embedding_size = embedding_size
        if embedding_size is not None:
            self.embedding_conv = ConvBNLayer(
                in_channels=self.out_channels,
                out_channels=embedding_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                if_act=False,
                name=prefix_name + 'conv_last_embedding')
            self.out_channels = embedding_size

        self.last_act = last_act

        if pool_kernel_size == 2:
            self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        else:
            # improved shape
            self.pool = nn.AvgPool2D(
                kernel_size=[2, pool_kernel_size],
                stride=[2, pool_kernel_size],
                padding=0)

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.conv1(x)

        out = {}
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
            stride = self.cfg[idx][-1]
            stride = stride[0] if isinstance(stride, (
                list,
                tuple, )) else stride
            if stride == 2:
                out["backbone_st{}".format(idx)] = x
        x = self.conv2(x)
        if self.embedding_size is not None:
            x = self.embedding_conv(x)

        if self.add_pre_act_layer:
            out["final_pre_act"] = x

        if self.last_act == "hard_swish":
            if paddle.__version__ == "2.0.0-rc1":
                x = F.activation.hard_swish(x)
            else:
                x = F.activation.hardswish(x)
        else:
            assert False, "not impl as now!!!"

        x = self.pool(x)
        out["final_output"] = x
        return out if self.return_feat_dict else x
