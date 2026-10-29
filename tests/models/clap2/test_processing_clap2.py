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
"""Testing suite for the CLAP2 processor."""

import tempfile
import unittest

import numpy as np

from transformers import AutoProcessor, Clap2FeatureExtractor, Clap2Processor, GPT2Tokenizer
from transformers.testing_utils import require_torch


@require_torch
class Clap2ProcessorTest(unittest.TestCase):
    def setUp(self):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", model_max_length=256)
        for name, token in [("eos_token", "<|eos|>"), ("pad_token", "<|pad|>"), ("bos_token", "<|bos|>")]:
            tokenizer.add_special_tokens({name: token})
        tokenizer.add_bos_token = tokenizer.add_eos_token = True
        tokenizer.update_post_processor()
        self.processor = Clap2Processor(feature_extractor=Clap2FeatureExtractor(), tokenizer=tokenizer)
        self.audio = np.random.default_rng(0).standard_normal(32000).astype(np.float32) * 0.1

    def test_tokenizer_wraps_text(self):
        input_ids = self.processor(text="a dog barking")["input_ids"]
        self.assertEqual(input_ids[0], 50259)
        self.assertEqual(input_ids[-1], 50257)
        self.assertEqual(input_ids[1:-1], [64, 3290, 48961])  # tiktoken r50k_base

    def test_text_and_audio(self):
        inputs = self.processor(
            text=["a dog barking", "rain"], audio=self.audio, sampling_rate=32000, padding=True, return_tensors="pt"
        )
        self.assertSetEqual(set(inputs), {"input_ids", "attention_mask", "input_features", "input_features_mask"})
        self.assertEqual(inputs["input_ids"][1, -1].item(), 50258)
        features = self.processor.feature_extractor(self.audio, sampling_rate=32000, return_tensors="pt")
        self.assertTrue(inputs["input_features"].equal(features["input_features"]))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.processor.save_pretrained(tmpdirname)
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Clap2Processor)
        self.assertEqual(
            processor(text="a dog barking")["input_ids"], self.processor(text="a dog barking")["input_ids"]
        )
        self.assertEqual(processor.feature_extractor.to_dict(), self.processor.feature_extractor.to_dict())
