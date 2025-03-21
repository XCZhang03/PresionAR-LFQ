"""This file contains the definition of the look-free residual quantizer."""

from typing import Mapping, Text, Tuple

import torch
from einops import rearrange, reduce

from modeling.quantizer.lookup_free import LookupFreeQuantizer
from modeling.quantizer.mutivariante_lfq import MultivariantLFQ


class ResidualLFQ(torch.nn.Module):
    def __init__(
            self,
            token_size: int = 10,
            num_quantizers: int = 2,
            variants: list[int] = [2,3],
            scales: Tuple[float, ...] = None,
            commitment_cost: float = 0.25,
            entropy_loss_weight: float = 0.1,
            entropy_loss_temperature: float = 0.01,
            entropy_gamma: float = 1.0,
    ):
        super().__init__()
        self.token_size = token_size
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        if scales:
            assert len(scales) == num_quantizers
            self.scales = scales
        else:
            self.scales = [2 ** -ind for ind in range(num_quantizers)]

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
                    entropy_gamma=entropy_gamma,
                    scale = self.scales[ind],
                    variants=variants[ind],
                )
            )

        self.quantizers = torch.nn.ModuleList(self.quantizers)
    
    def forward(self, z: torch.Tensor, num_levels: int=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
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
        
        if num_levels is None:
            num_levels = self.num_quantizers 
        else:
            assert num_levels <= self.num_quantizers
        for ind, quantizer in enumerate(self.quantizers[:num_levels]):
            z_quantized, result_dict = quantizer(residual)
            all_results.append(result_dict)
            quantized_out = quantized_out + z_quantized
            residual = residual - z_quantized.detach()
        
        all_result_dict = {}
        all_result_dict = {key: torch.stack([result_dict[key] for result_dict in all_results], dim=0) for key in all_results[0].keys()}

        # sum the losses
        all_result_dict["quantizer_loss"] = all_result_dict["quantizer_loss"].mean(dim=0)
        all_result_dict["commitment_loss"] = all_result_dict["commitment_loss"].mean(dim=0)
        all_result_dict["entropy_loss"] = all_result_dict["entropy_loss"].mean(dim=0)

        # ## debug the gradient
        # grad = torch.autograd.grad(quantized_out.sum(), z, create_graph=True)[0]


        ## STE estimator?
        quantized_out = z + (quantized_out - z).detach()

        return quantized_out, all_result_dict
    
    def get_codebook_entry(self, indices: torch.Tensor):
        """ Get the codebook entry for the given indices.

        Args:
            indices -> torch.Tensor: The indices of the codebook entry. shape: (n, ...)

        Returns:
            codebook_entry -> torch.Tensor: The codebook entry.
        """
        N, B, *_ = indices.shape
        assert N <= self.num_quantizers
        indices = torch.chunk(indices, chunks=N, dim=0)
        all_tokens = [quantizer.get_codebook_entry(index.squeeze(0)) for quantizer, index in zip(self.quantizers, indices)]
        return torch.stack(all_tokens, dim=0).sum(dim=0)



        
if __name__ == "__main__":
    quantizer = ResidualLFQ(num_quantizers=3)
    z = torch.randn(1, 10, 32, 32).requires_grad_()
    quantized, outputs = quantizer(z)
    for key, value in outputs.items():
        print(key, value.shape)
