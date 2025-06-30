import math
from typing import List, Tuple, Union, Optional
from copy import deepcopy
from omegaconf import ListConfig

import torch
from torch import nn

from modeling.cond_bert import CondBert, CondLFQBert
from modeling.bert import LFQBert, Bert
from modeling.modules import BaseModel

from utils.script_utils import get_model_kwargs

def get_stage_config(config, stage: int):
    cur_config = deepcopy(config)
    if 'mlm_model' not in cur_config.model or cur_config.model['mlm_model'] is None:
        cur_config.model['mlm_model'] = {}
    if stage == 0:
        cur_config.model['mlm_model'] = config.model.base_model
    else:
        # Ensure mlm_model exists and is a dict
        for key, value in config.model.cond_model.items():
            if isinstance(value, (ListConfig, list)):
                cur_config.model['mlm_model'][key] = value[stage - 1]
            else:
                cur_config.model['mlm_model'][key] = value
    cur_config.model.pop('cond_model', None)
    cur_config.model.pop('base_model', None)
    cur_config.model.pop('ar_model', None)
    return cur_config

class ResAR(BaseModel):
    def __init__(
            self,
            config,
    ):
        super(ResAR, self).__init__()
        self.num_stages = config.model.ar_model.num_stages
        self.config = config
        self.cur_stage = config.model.ar_model.get('cur_stage', None)
        self.configs = [get_stage_config(config, i) for i in range(self.num_stages)]
       
        model_cls = {
            "bert": Bert,
            "lfq_bert": LFQBert,
            "cond_bert": CondBert,
            "cond_lfq_bert": CondLFQBert,
        }
        self.base_model = model_cls[self.configs[0].model.mlm_model.model_cls](
            **get_model_kwargs(config=self.configs[0]),
        )
        self.configs[0].model.mlm_model.mask_token = self.base_model.mask_token
        
        self.cond_models = nn.ModuleList()
        for i in range(1, self.num_stages):
            model = model_cls[self.configs[i].model.mlm_model.model_cls](**get_model_kwargs(config=self.configs[i]))
            self.cond_models.append(model)
            self.configs[i].model.mlm_model.mask_token = model.mask_token

        if self.cur_stage is not None:
            self.set_stage(self.cur_stage)
        

    def set_stage(self, stage: int):
        assert 0 <= stage < self.num_stages, f"Stage must be between 0 and {self.num_stages - 1}, but got {stage}."
        if stage == 0:
            self.base_model.train()
            for model in self.cond_models:
                model.eval()
                model.requires_grad_(False)
        else:
            self.base_model.eval()
            self.base_model.requires_grad_(False)
            for i, model in enumerate(self.cond_models):
                if i == stage - 1:
                    model.train()
                    model.requires_grad_(True)
                else:
                    model.eval()
                    model.requires_grad_(False)

    @property
    def _cur_model(self):
        return self.get_stage_model(self.cur_stage) if self.cur_stage is not None else None
        
    @property
    def _cur_config(self):
        return self.configs[self.cur_stage] if self.cur_stage is not None else None
    
    def get_stage_model(self, stage: int):
        assert 0 <= stage < self.num_stages, f"Stage must be between 0 and {self.num_stages - 1}, but got {stage}."
        if stage == 0:
            return self.base_model
        else:
            return self.cond_models[stage - 1]

    def forward(self, *args, **kwargs):
        return self._cur_model(*args, **kwargs)



if __name__ == "__main__":
    device = "cuda"
    from omegaconf import OmegaConf
    config = OmegaConf.load("maskbit/configs/ar/ar_generator_10bit_4lvl.yaml")
    from modeling.rqgan import RQModel
    vqgan_model = RQModel(config.model.vq_model).to(device)
    config.model.vq_model.codebook_size = vqgan_model.codebook_size
    config.model.base_model.num_steps = 2
    config.model.cond_model.num_steps = 2
    model = ResAR(config).to(device)
    from modeling.modules.sampling import residual_sample
    samples = residual_sample(
        model=model,
        vqgan_model=vqgan_model,
        config=config,
    )