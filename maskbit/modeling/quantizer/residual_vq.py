"""This file contains the definition of the traditional vq residual quantizer."""

from typing import Mapping, Text, Tuple, List, Union

import torch
from einops import rearrange, reduce

from modeling.quantizer.quantizer import SimpleVectorizer
from modeling.quantizer.mutivariante_lfq import MultivariantLFQ
from modeling.quantizer.quant_scheduler import agg_quantized

from omegaconf import ListConfig

class ResidualVQ(torch.nn.Module):
    def __init__(
            self,
            codebook_size: Union[int, List[int]] = 1024,
            token_size: int = 256,
            num_quantizers: int = 4,
            commitment_cost: float = 0.25,
            entropy_loss_weight: float = 0.1,
            entropy_loss_temperature: float = 0.01,
            entropy_gamma: Union[float, List[float]] = 1.0,
            input_strides: Union[List[int], int] = 1,
            shared_codebook: bool = False,
            use_l2_normalisation: bool = False,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        if isinstance(entropy_gamma, (ListConfig, list)):
            assert len(entropy_gamma) == num_quantizers, "entropy_gamma should be a list of length num_quantizers"
        elif isinstance(entropy_gamma, float):
            entropy_gamma = [entropy_gamma] * num_quantizers
        else:
            raise ValueError("entropy_gamma should be a float or a list of floats")
        print(f"quantizer entropy_gamma: {entropy_gamma}")

        if isinstance(input_strides, (ListConfig, list)):
            assert len(input_strides) == num_quantizers, "input_strides should be a list of length num_quantizers"
            self.input_strides = input_strides
        elif isinstance(input_strides, int):
            input_strides = [input_strides] * num_quantizers
            self.input_strides = input_strides
        else:
            raise ValueError("input_strides should be an int or a list of ints")
        print(f"quantizer input_strides: {self.input_strides}")

        if isinstance(codebook_size, (ListConfig, list)):
            assert len(codebook_size) == num_quantizers, "codebook_size should be a list of length num_quantizers"
            self.codebook_size = codebook_size
        elif isinstance(codebook_size, int):
            self.codebook_size = [codebook_size] * num_quantizers
        else:
            raise ValueError("codebook_size should be an int or a list of ints")

        self.token_size = [token_size * (stride ** 2) for stride in input_strides]

        if shared_codebook:
            assert all(token_size == self.token_size[0]), "All token sizes must be equal when using a shared codebook"
            assert all(codebook_size == self.codebook_size[0]), "All codebook sizes must be equal when using a shared codebook"
            quantizer_0 = SimpleVectorizer(
                codebook_size=self.codebook_size[0],
                token_size=self.token_size[0],
                commitment_cost=commitment_cost,
                entropy_loss_weight=entropy_loss_weight,
                entropy_loss_temperature=entropy_loss_temperature,
                entropy_gamma=entropy_gamma[0],
                use_l2_normalisation=use_l2_normalisation,
            )
            self.quantizers = [quantizer_0] * num_quantizers
        else:
            self.quantizers = [
                SimpleVectorizer(
                    codebook_size=self.codebook_size[ind],
                    token_size=self.token_size[ind],
                    commitment_cost=commitment_cost,
                    entropy_loss_weight=entropy_loss_weight,
                    entropy_loss_temperature=entropy_loss_temperature,
                    entropy_gamma=entropy_gamma[ind],
                    use_l2_normalisation=use_l2_normalisation,
                )
                for ind in range(num_quantizers)
            ]
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
        assert all([(num_levels[ind] <= self.num_quantizers and num_levels[ind] >= 1) for ind in range(bs)])

        if loss_weight is None:
            loss_weight = [1] * self.num_quantizers  
        else:
            assert len(loss_weight) == self.num_quantizers, "loss_weight should be a list of length num_quantizers"
        loss_weight = torch.tensor(loss_weight, dtype=z.dtype, device=z.device).contiguous() 
        loss_weight = loss_weight / torch.sum(loss_weight)  # normalize the loss weight


        for ind, quantizer in enumerate(self.quantizers):
            ## patchify
            quant_input = rearrange(residual, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=self.input_strides[ind], p2=self.input_strides[ind])
            z_quantized, result_dict = quantizer(quant_input)
            ## unpatchify
            z_quantized = rearrange(z_quantized, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=self.input_strides[ind], p2=self.input_strides[ind])
            all_results.append(result_dict)
            quantized_list.append(z_quantized)
            # quantized_out = quantized_out + z_quantized
            residual = residual - z_quantized.detach()
        
        # aggregate the quantized tensors
        quantized_out = agg_quantized(quantized_list, num_levels)
        
        all_result_dict = {}
        all_result_dict['min_encoding_indices'] = [result_dict['min_encoding_indices'] for result_dict in all_results]
        all_result_dict.update({key: torch.stack([result_dict[key] for result_dict in all_results], dim=0) for key in all_results[0].keys() if key != "min_encoding_indices" })
        breakpoint()
        # sum the losses
        all_result_dict["quantizer_loss"] = (all_result_dict["quantizer_loss"] * loss_weight).sum(dim=0)
        all_result_dict["commitment_loss"] = (all_result_dict["commitment_loss"] * loss_weight).sum(dim=0)
        all_result_dict["entropy_loss"] = (all_result_dict["entropy_loss"] * loss_weight).sum(dim=0)
        all_result_dict["codebook_loss"] = (all_result_dict["codebook_loss"] * loss_weight).sum(dim=0)
        # ## debug the gradient
        # grad = torch.autograd.grad(quantized_out.sum(), z, create_graph=True)[0]


        ## STE estimator?
        quantized_out = z + (quantized_out - z).detach()

        return quantized_out, all_result_dict
    
    def get_codebook_entry(self, indices: Union[List, torch.Tensor], num_level: int = None) -> torch.Tensor:
        """ Get the codebook entry for the given indices.

        Args:
            indices -> torch.Tensor: The indices of the codebook entry. shape: (n, ...) or (...)
            num_level -> int: The level of quantization precision. If None, it will sum up all the codemaps from level 0.
            range: [0, num_quantizers - 1]

        Returns:
            codebook_entry -> torch.Tensor: The codebook entry.
        """
        if num_level is None:
            N = len(indices)
            assert N <= self.num_quantizers
            all_tokens = [self.get_codebook_entry(indices[i], num_level=i) for i in range(N)]
            return torch.stack(all_tokens, dim=0).sum(dim=0)
        else:
            assert num_level < self.num_quantizers, f"num_level should be less than {self.num_quantizers}, but got {num_level}"
            B, *_ = indices.shape
            tokens = self.quantizers[num_level].get_codebook_entry(indices)
            if len(tokens.shape) == 3:  # if tokens are flattened
                import math
                tokens = rearrange(tokens, "b (h w) (c p1 p2) -> b (h p1 w p2) c", h=int(math.sqrt(float(tokens.size(1)))), w=int(math.sqrt(float(tokens.size(1)))), p1=self.input_strides[num_level], p2=self.input_strides[num_level])
            else:
                tokens = rearrange(tokens, "b h w (c p1 p2) -> b (h p1) (w p2) c", p1=self.input_strides[num_level], p2=self.input_strides[num_level])
            return tokens
                
if __name__ == "__main__":
    quantizer = ResidualVQ(num_quantizers=3, input_strides=[2, 2, 1])
    z = torch.randn(3, 256, 32, 32).requires_grad_()
    quantized, outputs = quantizer(z, num_levels=3, loss_weight=[2,1,1])
    for key, value in outputs.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else len(value))
    z_3 = quantizer.get_codebook_entry(outputs['min_encoding_indices'][2], num_level=2)
    z2 = quantizer.get_codebook_entry(outputs['min_encoding_indices'][:2])
    z_hat = z2 + z_3
    assert torch.allclose(z_hat, quantized.permute(0, 2, 3, 1), atol=1e-5)  # check if the quantized output is equal to the codebook entry