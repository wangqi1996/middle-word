# Copyright 2017 Natural Language Processing Group, Nanjing University, zhengzx.142857@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from src.models.dl4mt import DL4MT
from src.models.dl4mt_1GRU import DL4MT_1GRU
from src.models.middle import Middle
from src.models.smart_start import SS_Transformer
from src.models.transformer import Transformer

__all__ = [
    "build_model",
]

MODEL_CLS = {
    "Transformer": Transformer,
    "DL4MT": DL4MT,
    'middle': Middle,
    'DL4MT_1GRU': DL4MT_1GRU,
    'SS_Transformer': SS_Transformer
}


def build_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)
