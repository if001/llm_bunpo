def get_config(model_name):
    if model_name not in name_to_config:
        raise ValueError('model not impl')
    conf_dict = name_to_config[model_name]
    return conf_dict

configs = []


################
# qwen2
################
qwen2 = [
    dict(
        name="qwen2-0.5B",
        hf_config=dict(org="qwen", name="qwen2-0.5B"),
        vocab_size=50257,
        attention_dropout=0.0,
        bos_token_id=1, ## llm-jp
        eos_token_id=7, ## llm-jp
        hidden_act="silu",
        hidden_size=896,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=131072,
        max_window_layers=24,
        model_type="qwen2",
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=131072,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        transformers_version="4.40.1",
        use_cache=True,
        use_sliding_window=False,
    ),
    dict(
        name="qwen2-0.1B",
        hf_config=dict(org="qwen", name="qwen2-0.1B"),
        vocab_size=50257,
        attention_dropout=0.0,
        bos_token_id=1, ## llm-jp
        eos_token_id=7, ## llm-jp
        hidden_act="silu",
        hidden_size=512,
        initializer_range=0.02,
        intermediate_size=2048,
        max_position_embeddings=131072,
        max_window_layers=24,
        model_type="qwen2",
        num_attention_heads=6,
        num_hidden_layers=6,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=131072,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        transformers_version="4.40.1",
        use_cache=True,
        use_sliding_window=False,
    ),
]
configs.extend(qwen2)

phi3 = [
    dict(
        name='phi3',
        vocab_size=50257,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    ),
    dict(
        name='phi3-small',
        vocab_size=50257,
        hidden_size=896,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    ),
    dict(
        name='phi3-tiny',
        vocab_size=50257,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=7,
        sliding_window=None,
    )
]
configs.extend(phi3)
name_to_config = {config["name"]: config for config in configs}