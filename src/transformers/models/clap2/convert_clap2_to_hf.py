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
"""Convert a CLAP2 OpenCLIP checkpoint (`open_clip_config.json` + `open_clip_model.safetensors`) to Transformers.

Usage:

```bash
python src/transformers/models/clap2/convert_clap2_to_hf.py --input_dir laion/clap2-large-audioxl-s1B --output_dir clap2-large
```
"""

import argparse
import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import (
    Clap2AudioConfig,
    Clap2Config,
    Clap2FeatureExtractor,
    Clap2Model,
    Clap2Processor,
    Clap2TextConfig,
    GPT2Tokenizer,
)


# Applied in order; fused projections are split afterwards in `convert_state_dict`.
KEY_MAPPING = [
    (r"^audio\.encoder\.vit\.embeds\.proj\.", "audio_model.embeddings.patch_embedding."),
    (r"^audio\.encoder\.vit\.embeds\.reg_token", "audio_model.embeddings.register_tokens"),
    (r"^audio\.encoder\.vit\.norm_pre\.", "audio_model.embeddings.layernorm."),
    (r"^audio\.encoder\.vit\.blocks\.", "audio_model.encoder.layers."),
    (r"^audio\.encoder\.vit\.norm\.", "audio_model.norm."),
    (r"^audio\.encoder\.vit\.attn_pool\.latent", "audio_model.head.probe"),
    (r"^audio\.encoder\.vit\.attn_pool\.proj\.", "audio_model.head.o_proj."),
    (r"^audio\.encoder\.vit\.attn_pool\.norm\.", "audio_model.head.layernorm."),
    (r"^audio\.encoder\.vit\.attn_pool\.mlp\.", "audio_model.head."),
    (r"^audio\.encoder\.vit\.attn_pool\.", "audio_model.head."),
    (r"^audio\.proj\.0\.", "audio_projection.linear1."),
    (r"^audio\.proj\.2\.", "audio_projection.linear2."),
    (r"^text\.token_embedding\.", "text_model.embed_tokens."),
    (r"^text\.blocks\.", "text_model.encoder.layers."),
    (r"^text\.ln_final\.", "text_model.norm."),
    (r"^text\.pool\.query", "text_model.head.probe"),
    (r"^text\.pool\.", "text_model.head."),
    (r"^text\.text_projection\.", "text_projection."),
    (r"\.norm1\.", ".input_layernorm."),
    (r"\.norm2\.", ".post_attention_layernorm."),
    (r"\.attn\.proj\.", ".self_attn.o_proj."),
    (r"\.attn\.", ".self_attn."),
    (r"\.gamma_1$", ".layer_scale1.lambda1"),
    (r"\.gamma_2$", ".layer_scale2.lambda1"),
    (r"\.head\.q\.", ".head.q_proj."),
    (r"\.mlp\.(fc2|w3)\.", ".mlp.down_proj."),
    (r"\.mlp\.fc1\.", ".mlp.gate_up_proj."),
]
# Fused tensors, split along dim 0
SPLITS = [
    (r"\.self_attn\.qkv\.", [".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj."]),
    (r"\.head\.kv\.", [".head.k_proj.", ".head.v_proj."]),
]


def convert_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        for pattern, replacement in KEY_MAPPING:
            key = re.sub(pattern, replacement, key)
        split = [(pattern, names) for pattern, names in SPLITS if re.search(pattern, key)]
        if key.endswith(".mlp.w12.weight"):  # the text SwiGLU gates its second half: move the gate first
            key, value = key.replace(".w12.", ".gate_up_proj."), torch.cat(value.chunk(2)[::-1])
        if split:
            ((pattern, names),) = split
            for name, chunk in zip(names, value.chunk(len(names))):
                converted[re.sub(pattern, name, key)] = chunk.clone()
        else:
            converted[key] = value
    return converted


def convert_config(model_cfg, state_dict):
    audio_cfg, text_cfg = model_cfg["audio_cfg"], model_cfg["text_cfg"]
    vit_cfg = audio_cfg["naflexvit_cfg"]
    assert audio_cfg["patch_freq"] == audio_cfg["patch_time"] and audio_cfg["rope_type"] == "axial"
    assert vit_cfg["swiglu_mlp"] and vit_cfg["norm_layer"] == "rmsnorm" and vit_cfg["pre_norm"]
    assert (text_cfg["pos_embed"], text_cfg["pool_type"], text_cfg["mlp_type"], text_cfg["norm_type"]) == (
        "rope",
        "map",
        "swiglu",
        "rmsnorm",
    )
    assert text_cfg["attention_mode"] == "bidirectional" and text_cfg["qk_norm"] and not text_cfg["attn_gated"]
    assert not text_cfg["attention_bias"] and not text_cfg["mlp_bias"] and text_cfg["norm_eps"] == 1e-6
    assert text_cfg["rope_temperature"] == 10000.0
    audio_config = Clap2AudioConfig(
        hidden_size=vit_cfg["embed_dim"],
        intermediate_size=state_dict["audio.encoder.vit.blocks.0.mlp.fc2.weight"].shape[1],
        num_hidden_layers=vit_cfg["depth"],
        num_attention_heads=vit_cfg["num_heads"],
        num_mel_bins=audio_cfg["mel_bins"],
        patch_size=audio_cfg["patch_freq"],
        num_register_tokens=vit_cfg["reg_tokens"],
        layerscale_value=vit_cfg["init_values"],
    )
    assert state_dict["audio.encoder.vit.attn_pool.mlp.fc1.weight"].shape[0] == audio_config.intermediate_size
    text_config = Clap2TextConfig(
        vocab_size=text_cfg["vocab_size"],
        hidden_size=text_cfg["width"],
        intermediate_size=state_dict["text.blocks.0.mlp.w3.weight"].shape[1],
        num_hidden_layers=text_cfg["layers"],
        num_attention_heads=text_cfg["heads"],
        max_position_embeddings=text_cfg["context_length"],
        pad_token_id=text_cfg["pad_id"],
        bos_token_id=text_cfg["bos_id"],
        eos_token_id=text_cfg["eos_id"],
    )
    return Clap2Config(text_config=text_config, audio_config=audio_config, projection_dim=model_cfg["embed_dim"])


def convert_processor(model_cfg):
    audio_cfg, text_cfg = model_cfg["audio_cfg"], model_cfg["text_cfg"]
    feature_extractor = Clap2FeatureExtractor(
        feature_size=audio_cfg["mel_bins"],
        sampling_rate=audio_cfg["sample_rate"],
        hop_length=audio_cfg["hop_size"],
        n_fft=audio_cfg["window_size"],
        f_min=float(audio_cfg["fmin"]),
        f_max=float(audio_cfg["fmax"]),
        patch_size=audio_cfg["patch_freq"],
    )
    # OpenCLIP tokenizes with tiktoken's `r50k_base` (the GPT-2 vocabulary) and wraps each text as
    # [<|bos|>] + text + [<|eos|>], with three special tokens appended after the vocabulary.
    assert text_cfg["tokenizer_type"] == "tiktoken" and text_cfg["tiktoken_name"] == "r50k_base"
    tokenizer = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2", model_max_length=text_cfg["context_length"], split_special_tokens=True
    )
    for name, token in [("eos_token", "<|eos|>"), ("pad_token", "<|pad|>"), ("bos_token", "<|bos|>")]:
        tokenizer.add_special_tokens({name: token})
    tokenizer.add_bos_token = tokenizer.add_eos_token = True
    tokenizer.update_post_processor()
    special_ids = (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id)
    assert special_ids == (text_cfg["eos_id"], text_cfg["pad_id"], text_cfg["bos_id"]), special_ids
    return Clap2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Local directory or Hub repo of the OpenCLIP checkpoint")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = args.input_dir if os.path.isdir(args.input_dir) else snapshot_download(args.input_dir)
    with open(os.path.join(input_dir, "open_clip_config.json"), encoding="utf-8") as f:
        model_cfg = json.load(f)["model_cfg"]
    state_dict = load_file(os.path.join(input_dir, "open_clip_model.safetensors"))

    model = Clap2Model(convert_config(model_cfg, state_dict))
    model.load_state_dict(convert_state_dict(state_dict), strict=True)
    model.save_pretrained(args.output_dir)
    convert_processor(model_cfg).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
