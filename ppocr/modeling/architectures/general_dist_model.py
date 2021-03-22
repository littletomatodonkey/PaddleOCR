# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head
from .base_model import BaseModel

__all__ = ['GeneralDistModel']


class GeneralDistModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(GeneralDistModel, self).__init__()

        # assert isinstance(config["Teacher"], list), "teeacher models must be a list, but got {}".format(type(config["Teacher"]))

        self.build_models(config)

        self.freeze_teacher = config.get("freeze_teacher", True)

        if self.freeze_teacher:
            for teacher in self.teacher:
                for param in teacher.parameters():
                    param.trainable = False
        else:
            print(
                "teacher model is not freezed during during training process...")

    def build_models(self, config):
        # build teacher models
        self.teacher = []
        for key in config["Teacher"]:
            teacher_config = config["Teacher"][key]
            print(teacher_config)
            teacher_config["model_type"] = config["model_type"]
            teacher_config["algorithm"] = config["algorithm"]
            # add sublayer is needed to update model
            self.teacher.append(
                self.add_sublayer(key, BaseModel(teacher_config)))

        config["Student"]["model_type"] = config["model_type"]
        config["Student"]["algorithm"] = config["algorithm"]
        self.student = BaseModel(config["Student"])

    def forward(self, x):

        teacher_list_out = []
        for teacher in self.teacher:
            out = teacher.forward(x, fetch_all_feats=True)
            teacher_list_out.append(out)

        student_out = self.student.forward(x, fetch_all_feats=True)

        result = {
            "teacher_list_out": teacher_list_out,
            "student_out": student_out,
        }
        return result
