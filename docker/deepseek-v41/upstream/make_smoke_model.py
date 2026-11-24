"""Create a bounded synthetic checkpoint for an isolated SM120 SGLang HTTP smoke."""

import argparse
import json
from pathlib import Path


def make_tokenizer(target, template):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    if template:
        actual = json.loads((template.parent / "tokenizer.json").read_text())
        reserved = [
            item["content"] for item in actual["added_tokens"]
            if "place" not in item["content"] and "unused" not in item["content"]
        ]
        bos, eos, pad = reserved[:3]
        words = [bos, eos, pad, "[UNK]", "ping", "ready", *reserved[3:]]
    else:
        pad, bos, eos = "[PAD]", "[BOS]", "[EOS]"
        reserved = [pad, bos, eos]
        words = [pad, bos, eos, "[UNK]", "ping", "ready"]
    if len(words) > 64:
        raise ValueError("DeepSeek reserved vocabulary exceeds the smoke checkpoint size")
    words.extend(f"word{i}" for i in range(64 - len(words)))
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(64))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(reserved)
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=pad, bos_token=bos, eos_token=eos,
        unk_token="[UNK]",
    ).save_pretrained(target)
    return tokenizer.token_to_id("<｜deepseek_image｜>")



def llama(target):
    import torch
    from safetensors.torch import save_file

    config = dict(
        architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=256,
        head_dim=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        intermediate_size=384, hidden_act="silu", max_position_embeddings=256,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, vocab_size=64, dtype="bfloat16",
    )
    torch.manual_seed(1017)

    def random_weight(shape):
        return (torch.randn(shape) * 0.001).bfloat16()

    weights = {"model.embed_tokens.weight": random_weight((64, 256))}
    # Make ping and ready predict ready. Every transformer layer has nonzero weights.
    weights["model.embed_tokens.weight"][4:6, 0] = 2
    weights["model.norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
    weights["lm_head.weight"] = random_weight((64, 256))
    weights["lm_head.weight"][5, 0] = 2
    shapes = {
        "self_attn.q_proj": (256, 256), "self_attn.k_proj": (128, 256),
        "self_attn.v_proj": (128, 256), "self_attn.o_proj": (256, 256),
        "mlp.gate_proj": (384, 256), "mlp.up_proj": (384, 256),
        "mlp.down_proj": (256, 384),
    }
    for index in range(2):
        prefix = f"model.layers.{index}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + norm + ".weight"] = torch.ones(256, dtype=torch.bfloat16)
        for name, shape in shapes.items():
            weights[prefix + name + ".weight"] = random_weight(shape)
    save_file(weights, str(target / "model.safetensors"), metadata={"format": "pt"})
    return config


def deepseek(template, image_token):
    # --load-format dummy initializes weights through the native quantized loader.
    config = json.loads(template.read_text())
    if config["model_type"] != "deepseek_v41":
        raise ValueError("Supply a DeepSeek-V4.1 config.json")
    config.pop("vision_config", None)
    config["image_token_id"] = image_token
    text = config["text_config"]
    text.update(
        hidden_size=512, n_routed_experts=4, n_shared_experts=1,
        num_hidden_layers=2, num_attention_heads=16, index_n_heads=16,
        q_lora_rank=256, o_lora_rank=128, o_groups=4, moe_intermediate_size=256,
        num_experts_per_tok=2, vocab_size=64, max_position_embeddings=1024,
        dspark_noise_token_id=55, kv_source_layer_ids=[1], index_source_layer_ids=[1],
        candidate_source_layer_id=-1, candidate_topk_blocks=0, candidate_block_size=8,
        compress_ratios=[0, 2], engram_layer_ids=[], engram_num_embeddings=[],
        dspark_target_layer_ids=[], num_nextn_predict_layers=0,
    )
    text["rope_scaling"]["original_max_position_embeddings"] = 64
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="A new directory to create")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--llama", action="store_true")
    model.add_argument("--deepseek-config", type=Path, help="V4.1-Flash 1017 config.json")
    parser.add_argument("--pearl-projection", action="store_true", help="Match the local 2048-to-5120 checkpoint projection at TP1")
    args = parser.parse_args()
    if args.pearl_projection and not args.deepseek_config:
        parser.error("--pearl-projection requires --deepseek-config")
    args.target.mkdir(parents=True)
    image_token = make_tokenizer(args.target, args.deepseek_config)
    config = llama(args.target) if args.llama else deepseek(args.deepseek_config, image_token)
    if args.pearl_projection:
        config["text_config"].update(hidden_size=5120, o_lora_rank=512, o_groups=4)
    (args.target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
