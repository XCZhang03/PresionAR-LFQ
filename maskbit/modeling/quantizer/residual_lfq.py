"""This file contains the definition of the look-free residual quantizer."""

from typing import Mapping, Text, Tuple, List, Union

import torch
from einops import rearrange, reduce

from modeling.quantizer.lookup_free import LookupFreeQuantizer
from modeling.quantizer.mutivariante_lfq import MultivariantLFQ
from modeling.quantizer.quant_scheduler import agg_quantized

from omegaconf import ListConfig
class ResidualLFQ(torch.nn.Module):
    def __init__(
            self,
            token_size: int = 10,
            num_quantizers: int = 2,
            variants: List[int] = [2,3],
            scales:  Union[List[int], int] = None,
            commitment_cost: float = 0.25,
            entropy_loss_weight: float = 0.1,
            entropy_loss_temperature: float = 0.01,
            entropy_gamma: Union[float, List[float]] = 1.0,
    ):
        super().__init__()
        self.token_size = token_size
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        assert num_quantizers == len(variants)

        if isinstance(scales, (ListConfig, list)):
            assert len(scales) == num_quantizers
            self.scales = scales
        elif isinstance(scales, int):
            self.scales = [scales ** -ind for ind in range(num_quantizers)]
        else:
            self.scales = [2 ** -ind for ind in range(num_quantizers)]
        print(f"quantizer scales: {self.scales}")

        if isinstance(entropy_gamma, (ListConfig, list)):
            assert len(entropy_gamma) == num_quantizers, "entropy_gamma should be a list of length num_quantizers"
        elif isinstance(entropy_gamma, float):
            entropy_gamma = [entropy_gamma] * num_quantizers
        else:
            raise ValueError("entropy_gamma should be a float or a list of floats")
        print(f"quantizer entropy_gamma: {entropy_gamma}")

        self.quantizers = []
        # self.quantizers.append(
        #     LookupFreeQuantizer(
        #         token_bits=token_size,
        #         commitment_cost=commitment_cost,
        #         entropy_loss_weight=entropy_loss_weight,
        #         entropy_loss_temperature=entropy_loss_temperature,
        #         entropy_gamma=entropy_gamma,
        #     )
        # )
        for ind in range(num_quantizers):
            self.quantizers.append(
                MultivariantLFQ(
                    token_size=token_size,
                    commitment_cost=commitment_cost,
                    entropy_loss_weight=entropy_loss_weight,
                    entropy_loss_temperature=entropy_loss_temperature,
                    entropy_gamma=entropy_gamma[ind],
                    scale = self.scales[ind],
                    variants=variants[ind],
                )
            )

        self.quantizers = torch.nn.ModuleList(self.quantizers)
    
    def forward(self, z: torch.Tensor, num_levels: Union[List,int]=None, loss_weight: List[int]=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Forward pass of the quantizer.

        Args:
            z -> torch.Tensor: The input tensor. shape: (b, c, h, w)
            num_levels -> int: The number of levels to quantize the input tensor to. range: [1, num_quantizers]

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        quantized_out = 0
        residual = z

        all_results = []
        quantized_list = []
        bs = z.shape[0]
        
        if num_levels is None:
            num_levels = self.num_quantizers 
        if isinstance(num_levels, int):
            num_levels = [num_levels] * bs
        assert isinstance(num_levels, list)
        assert len(num_levels) == bs
        assert all([(num_levels[ind] <= self.num_quantizers and num_levels[ind] >= 0) for ind in range(bs)])

        if loss_weight is None:
            loss_weight = [1] * self.num_quantizers  
        else:
            assert len(loss_weight) == self.num_quantizers, "loss_weight should be a list of length num_quantizers"
        loss_weight = torch.tensor(loss_weight, dtype=z.dtype, device=z.device).contiguous() 
        loss_weight = loss_weight / torch.sum(loss_weight)  # normalize the loss weight


        for ind, quantizer in enumerate(self.quantizers):
            z_quantized, result_dict = quantizer(residual)
            all_results.append(result_dict)
            quantized_list.append(z_quantized)
            # quantized_out = quantized_out + z_quantized
            residual = residual - z_quantized.detach()
        
        # aggregate the quantized tensors
        quantized_out = agg_quantized(quantized_list, num_levels)
        
        all_result_dict = {}
        all_result_dict = {key: torch.stack([result_dict[key] for result_dict in all_results], dim=0) for key in all_results[0].keys()}

        # sum the losses
        all_result_dict["quantizer_loss"] = (all_result_dict["quantizer_loss"] * loss_weight).sum(dim=0)
        all_result_dict["commitment_loss"] = (all_result_dict["commitment_loss"] * loss_weight).sum(dim=0)
        all_result_dict["entropy_loss"] = (all_result_dict["entropy_loss"] * loss_weight).sum(dim=0)
        # ## debug the gradient
        # grad = torch.autograd.grad(quantized_out.sum(), z, create_graph=True)[0]


        ## STE estimator?
        quantized_out = z + (quantized_out - z).detach()

        ## zero out the gradients with zero levels
        mask = torch.zeros(bs, dtype=torch.bool).to(z.device)
        for b in range(bs):
            mask[b] = num_levels[b] > 0
        quantized_out = quantized_out * mask.view(bs, *((1,) * (len(z.shape) - 1)))

        return quantized_out, all_result_dict
    
    def get_codebook_entry(self, indices: torch.Tensor, num_level: int = None) -> torch.Tensor:
        """ Get the codebook entry for the given indices.

        Args:
            indices -> torch.Tensor: The indices of the codebook entry. shape: (n, ...) or (...)
            num_level -> int: The level of quantization precision. If None, it will sum up all the codemaps from level 0.
            range: [0, num_quantizers - 1]

        Returns:
            codebook_entry -> torch.Tensor: The codebook entry.
        """
        if num_level is None:
            N, B, *_ = indices.shape
            assert N <= self.num_quantizers
            indices = torch.chunk(indices, chunks=N, dim=0)
            all_tokens = [quantizer.get_codebook_entry(index.squeeze(0)) for quantizer, index in zip(self.quantizers, indices)]
            return torch.stack(all_tokens, dim=0).sum(dim=0)
        else:
            assert num_level < self.num_quantizers, f"num_level should be less than {self.num_quantizers}, but got {num_level}"
            B, *_ = indices.shape
            tokens = self.quantizers[num_level].get_codebook_entry(indices)
            return tokens


        
if __name__ == "__main__":
    quantizer = ResidualLFQ(num_quantizers=3, variants=[2,3,3], scales=4)
    z = torch.randn(3, 10, 32, 32).requires_grad_()
    quantized, outputs = quantizer(z, num_levels=3, loss_weight=[2,1,1])
    for key, value in outputs.items():
        print(key, value.shape)
    z_3 = quantizer.get_codebook_entry(outputs['min_encoding_indices'][2], num_level=2)
    z2 = quantizer.get_codebook_entry(outputs['min_encoding_indices'][:2])
    z_hat = z2 + z_3
    assert torch.allclose(z_hat, quantized.permute(0, 2, 3, 1), atol=1e-5)  # check if the quantized output is equal to the codebook entry
