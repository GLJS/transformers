# Copyright 2026 LAION and The HuggingFace Team. All rights reserved.
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
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class Clap2AudioKwargs(AudioKwargs, total=False):
    """
    max_num_patches (`int`, *optional*):
        Longer clips keep only their first time columns of patches that fit in `max_num_patches`. Defaults to the
        `max_num_patches` of the feature extractor.
    """

    max_num_patches: int


class Clap2ProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Clap2AudioKwargs
    _defaults = {}


@auto_docstring
class Clap2Processor(ProcessorMixin):
    valid_processor_kwargs = Clap2ProcessorKwargs

    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)


__all__ = ["Clap2Processor"]
