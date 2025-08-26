import torch
import random


class PostEncAug(torch.nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.config = config
        self.setup()
        self.disable_during_inference = self.config.get("disable_during_inference", False)
        self.disabled = False
    
    def setup(self):
        self.std = self.config.get("std", 0.1) ## gamma
        self.scale = self.config.get("scale", 1.0)  ## tau
        self.same_over_channel = self.config.get("same_over_channel", False)
        if self.config.get("noise_mode", None) is None:
            self.disabled = True
            return
        if self.config.noise_mode == "add":
            def noise_fn(x: torch.Tensor):
                B, C, H, W = x.shape
                if self.same_over_channel:
                    noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype) * self.std
                    noise = noise.expand_as(x)
                else:
                    noise = torch.randn(B, C, H, W, device=x.device, dtype=x.dtype) * self.std
                return x + noise
        elif self.config.noise_mode == "interpolate":
            def noise_fn(x: torch.Tensor):
                scale = random.uniform(0, self.scale)
                B, C, H, W = x.shape
                if self.same_over_channel:
                    noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype) * self.std
                    noise = noise.expand_as(x)
                else:
                    noise = torch.randn(B, C, H, W, device=x.device, dtype=x.dtype) * self.std
                return x * (1 - scale) + noise * scale
        else:
            raise NotImplementedError(f"Unknown noise mode {self.config.noise_mode}")
        self.noise_fn = noise_fn
        return 


        

    def forward(self, x: torch.Tensor):
        if self.disabled or (not self.training and self.disable_during_inference):
            return x
        return self.noise_fn(x)