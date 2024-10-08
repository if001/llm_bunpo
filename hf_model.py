from transformers import (
    Phi3ForCausalLM, 
    Phi3Config,
    Qwen2ForCausalLM, 
    Qwen2Config
)

class Phi3(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(Phi3Config(**config))

class Qwen2(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(Qwen2Config(**config))

def get_hf_models(config):
    if 'name' not in config:
        raise ValueError('config must have name field')
    model_name = config['name']
    if 'phi3' in model_name:
        return Phi3(config)
    elif 'qwen2' in model_name:
        return Qwen2(config)
    else:
        raise ValueError('not impl hf models: ', model_name)