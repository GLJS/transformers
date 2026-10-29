<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-29.*

# CLAP2

## Overview

CLAP2 is a family of contrastive audio-text models from LAION, trained on LAION-Audio-XL. It embeds audio clips and texts in a shared space for zero-shot audio classification and audio-text retrieval.

The audio encoder is a NaFlex vision transformer on log-mel patches: a clip of any length becomes a sequence of 16x16 patches of a 128-bin log-mel spectrogram, with 2D axial rotary position embeddings over frequency and time, a register token, SwiGLU MLPs with layer scale, and attention pooling. The text encoder is a bidirectional transformer over the GPT-2 vocabulary with rotary position embeddings, QK normalization and attention pooling.

The checkpoints come in five sizes: [xxs](https://huggingface.co/laion/clap2-xxs-audioxl-s1B), [xs](https://huggingface.co/laion/clap2-xs-audioxl-s1B), [small](https://huggingface.co/laion/clap2-small-audioxl-s1B), [medium](https://huggingface.co/laion/clap2-medium-audioxl-s1B) and [large](https://huggingface.co/laion/clap2-large-audioxl-s1B).

This model was contributed by [LAION](https://huggingface.co/laion). The original code can be found in [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Usage

```py
import torch
from datasets import Audio, load_dataset
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("laion/clap2-large-audioxl-s1B")
processor = AutoProcessor.from_pretrained("laion/clap2-large-audioxl-s1B")

dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))
audio = next(iter(dataset))["audio"]["array"]
texts = ["the sound of a dog", "the sound of rain", "the sound of a vacuum cleaner"]

inputs = processor(text=texts, audio=audio, sampling_rate=32000, padding=True, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)
print(dict(zip(texts, outputs.logits_per_audio.softmax(dim=-1)[0].tolist())))
```

## Usage tips

- The feature extractor expects mono audio at 32 kHz and does not resample. Average the channels and resample beforehand, for example with `datasets.Audio(sampling_rate=32000)`.
- Clips are not cropped. A clip becomes `8 x ceil(num_frames / 16)` patches and a batch is padded to its longest clip, with `input_features_mask` marking the real patches. Set `max_num_patches` on the feature extractor to cap long clips (`max_num_patches=1504` keeps the first 30 seconds).
- OpenCLIP pads every text to 256 tokens. `padding="max_length"` reproduces that, while `padding=True` gives the same embeddings up to float32 rounding.

## Clap2Config

[[autodoc]] Clap2Config

## Clap2TextConfig

[[autodoc]] Clap2TextConfig

## Clap2AudioConfig

[[autodoc]] Clap2AudioConfig

## Clap2FeatureExtractor

[[autodoc]] Clap2FeatureExtractor
    - __call__

## Clap2Processor

[[autodoc]] Clap2Processor

## Clap2Model

[[autodoc]] Clap2Model
    - forward
    - get_text_features
    - get_audio_features

## Clap2TextModel

[[autodoc]] Clap2TextModel
    - forward

## Clap2AudioModel

[[autodoc]] Clap2AudioModel
    - forward
