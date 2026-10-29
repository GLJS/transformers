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
"""Testing suite for the CLAP2 feature extractor."""

import tempfile
import unittest

import numpy as np

from transformers import Clap2FeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio


@require_torch
class Clap2FeatureExtractionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # 0.1 s (less than one patch), 1 s and 2.5 s of noise at 32 kHz
        self.clips = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (3200, 32000, 80000)]

    def test_shapes_and_mask(self):
        feature_extractor = Clap2FeatureExtractor()
        inputs = feature_extractor(self.clips, sampling_rate=32000, return_tensors="pt")
        # 1 + samples // 320 frames, rounded up to whole patches of 16 frames, times 8 frequency patches
        num_patches = [8 * -(-(1 + len(clip) // 320) // 16) for clip in self.clips]
        self.assertEqual(tuple(inputs["input_features"].shape), (3, max(num_patches), 256))
        self.assertListEqual(inputs["input_features_mask"].sum(-1).tolist(), num_patches)
        self.assertTrue((inputs["input_features"][0, num_patches[0] :] == 0).all())

    def test_batched_matches_single(self):
        feature_extractor = Clap2FeatureExtractor()
        batched = feature_extractor(self.clips, sampling_rate=32000, return_tensors="pt")
        for i, clip in enumerate(self.clips):
            single = feature_extractor(clip, sampling_rate=32000, return_tensors="pt")["input_features"][0]
            self.assertTrue(batched["input_features"][i, : len(single)].equal(single))

    def test_max_num_patches(self):
        feature_extractor = Clap2FeatureExtractor(max_num_patches=20)
        inputs = feature_extractor(self.clips[2], sampling_rate=32000, return_tensors="pt")
        # whole time columns of 8 patches are kept
        self.assertEqual(inputs["input_features"].shape[1], 16)

    def test_save_load(self):
        feature_extractor = Clap2FeatureExtractor(max_num_patches=1504)
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor.save_pretrained(tmpdirname)
            loaded = Clap2FeatureExtractor.from_pretrained(tmpdirname)
        self.assertEqual(loaded.to_dict(), feature_extractor.to_dict())
        self.assertTrue(loaded.mel_filters.equal(feature_extractor.mel_filters))

    def test_wrong_sampling_rate_and_channels(self):
        feature_extractor = Clap2FeatureExtractor()
        with self.assertRaises(ValueError):
            feature_extractor(self.clips[1], sampling_rate=48000)
        with self.assertRaises(ValueError):
            feature_extractor([np.stack([self.clips[1], self.clips[1]])], sampling_rate=32000)

    @require_torchaudio
    def test_matches_torchaudio(self):
        import torch
        import torchaudio

        feature_extractor = Clap2FeatureExtractor()
        clip = self.clips[1]
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, hop_length=320, f_min=50, f_max=16000, n_mels=128, power=2.0, norm=None
        )(torch.from_numpy(clip))
        log_mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel).T
        num_time_patches = -(-log_mel.shape[0] // 16)
        fill = log_mel.min().expand(num_time_patches * 16 - log_mel.shape[0], 128)
        log_mel = torch.cat([log_mel, fill])
        expected = log_mel.reshape(num_time_patches, 16, 8, 16).permute(2, 0, 3, 1).reshape(-1, 256)
        patches = feature_extractor(clip, sampling_rate=32000, return_tensors="pt")["input_features"][0]
        torch.testing.assert_close(patches, expected, rtol=0, atol=0)
