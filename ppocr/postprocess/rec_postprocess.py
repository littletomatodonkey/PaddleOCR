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
from paddle.nn import functional as F


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'en_sensitive', 'french', 'german', 'japan', 'korean'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type in ["ch", "french", "german", "japan", "korean"]:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is ch"
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            import string
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """ convert text-index into text-label. """
        result_list = []
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
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, dict):
            preds = preds["head_out"]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 infer_teacher=False,
                 **kwargs):
        super(DistillationCTCLabelDecode, self).__init__(
            character_dict_path, character_type, use_space_char)
        self.infer_teacher = infer_teacher

    def __call__(self, preds, label=None, *args, **kwargs):
        # some exp when in 
        # pred = (preds["student_out"]["head_out"] + preds["teacher_out"]["head_out"]) / 2.0
        if self.infer_teacher:
            key = "teacher_list_out" if "teacher_list_out" in preds else "teacher_out"
            pred = preds[key]
            if isinstance(pred, dict):
                pred = pred["head_out"]
            elif isinstance(pred, list):
                ps = []
                for p in pred:
                    ps.append(p["head_out"])
                pred = paddle.add_n(ps) / len(ps)
            else:
                assert False, "pred type({}) wrong...".format(type(pred))
        else:
            pred = preds["student_out"]
        return super().__call__(pred, label=label, *args, **kwargs)

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              character_type, use_space_char)
        self.beg_str = "sos"
        self.end_str = "eos"

    def add_special_char(self, dict_character):
        dict_character = [self.beg_str, self.end_str] + dict_character
        return dict_character

    def __call__(self, text):
        text = self.decode(text)
        return text

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
