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
    if model_name == 'phi-3':
        return Phi3(config)
    elif model_name == 'qwen2':
        return Qwen2(config)
    else:
        raise ValueError('not impl hf models: ', model_name)