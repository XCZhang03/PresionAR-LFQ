"""This file contains the definition of the our tokenizer, which can use VQ or LFQ."""

import math
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange

from modeling.modules import (BaseModel, ConvDecoder, ConvDecoderLegacy,
                              ConvEncoder)
from modeling.quantizer import LookupFreeQuantizer, SimpleVectorizer

from modeling.quantizer import ResidualVQ, ResidualLFQ
from modeling.modules.autoencoder import ResidualStage, Conv2dSame
from typing import List, Union
from collections.abc import Iterable
from copy import deepcopy

def choose_vector_quantizer_class(config):
    if config.quantizer_type == "lookup":
        return SimpleVectorizer(
            config.codebook_size,
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
            config.get("use_l2_normalisation", False),
        )
    elif config.quantizer_type == "lookup-free":
        return LookupFreeQuantizer(
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
        )
    elif config.quantizer_type == "vae":
        return NotImplementedError("Currently not supported. We welcome a well tested PR.")
    else:
        raise ValueError("Unknown vector quantizer class")


class ConvVQModel(BaseModel):
    def __init__(
        self,
        config,
        legacy: bool = False,
        finetune_decoder: bool = False
    ):
        """ Initializes the convolutional VQ-VAE model.

        Args:
            config: The configuration for the model.
            legacy -> bool: Whether to use the legacy decoder, which is a different implementation of the same architecture.
            finetune_decoder -> bool: Whether to finetune the decoder.
        """
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(self.config)
        if legacy:
            # To support older weights and MaskGIT
            self.decoder = ConvDecoderLegacy(self.config)
        else:
            self.decoder = ConvDecoder(self.config)

        self.finetune_decoder = finetune_decoder
        if self.finetune_decoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        self.quantize = choose_vector_quantizer_class(self.config)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Encodes the input tensor, i.e. runs the encoder.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = self.encoder(x)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Decodes the quantized latent representation, i.e. runs the decoder.

        Args:
            z_quantized -> torch.Tensor: The quantized latent representation.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        decoded = self.decoder(z_quantized)
        return decoded

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        z_quantized = self.quantize.get_codebook_entry(tokens)
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the model on the input tensor.

        Args:
            input -> torch.Tensor: The input image.

        Returns:
            decoded -> torch.Tensor: The decoded image.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        if self.finetune_decoder:
            self.encoder.eval()
            z_quantized, result_dict = self._finetuning_encoder_forward(input)
        else:
            z_quantized, result_dict = self.encode(input)

        decoded = self.decode(z_quantized)
        return decoded, result_dict

    def _finetuning_encoder_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the encoder on the input tensor without gradients and sets quantizer losses to 0.

        Args:
            input -> torch.Tensor: The input image.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        with torch.no_grad():
            z_quantized, result_dict = self.encode(input)
            result_dict["quantizer_loss"] *= 0
            result_dict["commitment_loss"] *= 0
            if "codebook_loss" in result_dict:
                result_dict["codebook_loss"] *= 0
            result_dict["entropy_loss"] *= 0
        return z_quantized, result_dict
    

def choose_residual_vector_quantizer_class(config):
    if config.residual_quantizer_type == "residual_lfq":
        return ResidualLFQ(
            config.token_size,
            config.num_quantizers,
            config.variants,
            config.scales,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
        )
    elif config.residual_quantizer_type == "residual_vq":
        return ResidualVQ(
            config.codebook_size,
            config.token_size,
            config.num_quantizers,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
            config.get("input_strides", 1),
            config.get("shared_codebook", False),
            config.get("use_l2_normalisation", False),
        )


class FTConvVQModel(ConvVQModel):
    def __init__(self, config, legacy = False, finetune_decoder: bool = True):
        super().__init__(config, legacy, finetune_decoder=True)
        
        residual_quantizer_config = deepcopy(config)
        ## refactor the config for residual quantization
        if residual_quantizer_config.residual_quantizer_type == "residual_lfq":
            residual_quantizer_config.variants = residual_quantizer_config.variants[1:]
            residual_quantizer_config.scales = residual_quantizer_config.scales[1:] if residual_quantizer_config.scales is not None else None
            residual_quantizer_config.num_quantizers -= 1
        elif residual_quantizer_config.residual_quantizer_type == "residual_vq":
            residual_quantizer_config.num_quantizers -= 1
            residual_quantizer_config.codebook_size = residual_quantizer_config.codebook_size[1:] if isinstance(residual_quantizer_config.codebook_size, Iterable) else residual_quantizer_config.codebook_size
            residual_quantizer_config.input_strides = residual_quantizer_config.input_strides[1:] if isinstance(residual_quantizer_config.input_strides, Iterable) else residual_quantizer_config.input_strides
        self.residual_quantize = choose_residual_vector_quantizer_class(residual_quantizer_config)
        
        if config.pre_conv:
            self.pre_conv = torch.nn.Sequential(
                Conv2dSame(config.token_size, config.hidden_channels, kernel_size=3, bias=False),
                ResidualStage(config.hidden_channels, config.hidden_channels, config.num_res_blocks),
                Conv2dSame(config.hidden_channels, config.token_size, kernel_size=3, bias=False),
            )
        else:
            self.pre_conv = None
        self.num_quantizers = self.residual_quantize.num_quantizers + 1
        self.codebook_size = [self.quantize.codebook_size] + [q.codebook_size for q in self.residual_quantize.quantizers]
        

    def encode(self, x: torch.Tensor, num_levels: Union[List,int]=None, loss_weight: List[int]=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Encodes the input tensor, i.e. runs the encoder.

        Args:
            x -> torch.Tensor: The input tensor.
            num_levels -> int: The levels of quantization precision.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = self.encoder(x)
        base_quantized, result_dict = self.quantize(z)

        ## refactor the scheduler for residual quantization
        if isinstance(num_levels, List):
            num_levels = [i - 1 for i in num_levels]
        elif isinstance(num_levels, int):
            num_levels = num_levels - 1
        if isinstance(loss_weight, List):
            loss_weight = loss_weight[1:]
            loss_weight = [i / sum(loss_weight) for i in loss_weight]

        residual = z - base_quantized
        
        if self.pre_conv is not None:
            residual = self.pre_conv(residual)

        res_quantized, res_result_dict = self.residual_quantize(residual, num_levels=num_levels, loss_weight=loss_weight)
        
        if isinstance(res_result_dict['min_encoding_indices'], torch.Tensor):
            res_result_dict['min_encoding_indices'] = torch.cat([result_dict['min_encoding_indices'].unsqueeze(0), res_result_dict['min_encoding_indices']], dim=0)
        elif isinstance(res_result_dict['min_encoding_indices'], List):
            res_result_dict['min_encoding_indices'] = [result_dict['min_encoding_indices']] + res_result_dict['min_encoding_indices']
        else:
            raise ValueError("Unknown type of min_encoding_indices")
        
        z_quantized = base_quantized + res_quantized

        return z_quantized, res_result_dict
        
    def forward(self, input: torch.Tensor, num_levels: Union[List,int]=None, loss_weight: List[int]=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the model on the input tensor.

        Args:
            input -> torch.Tensor: The input image.
            num_levels -> int: The levels of quantization precision.
            loss_weight -> List[int]: The weights for the codebook loss of each quantizer.

        Returns:
            decoded -> torch.Tensor: The decoded image.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z_quantized, result_dict = self.encode(input,num_levels=num_levels, loss_weight=loss_weight)
        decoded = self.decode(z_quantized)
        return decoded, result_dict

    def decode_tokens(self, tokens: torch.Tensor, num_level: int=None, context: torch.Tensor=None) -> torch.Tensor:
        """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

        Args:
            tokens -> torch.Tensor: The token indices. shape: (n, b, h, w) or (n, b, h*w)

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        z_quantized = self.get_codebook_entry(tokens, num_level=num_level)
        if context is not None:
            assert num_level is not None, "only decode one level when context is provided"
            z_quantized = z_quantized + context
        decoded = self.decode(z_quantized)
        return decoded

    def get_codebook_entry(self, tokens: torch.Tensor, num_level: int=None) -> torch.Tensor:
        """ Gets the codebook entry for the given tokens.

        Args:
            tokens -> torch.Tensor: The token indices. shape: (n, b, h, w) or (n, b, h*w)

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
        """
        ## if num_level is 0 we only use the base quantizer
        if num_level is None:
            base_quantized = self.quantize.get_codebook_entry(tokens[0])
            residual_quantized = self.residual_quantize.get_codebook_entry(tokens[1:], num_level=None)
            z_quantized = base_quantized + residual_quantized
        elif num_level == 0:
            z_quantized = self.quantize.get_codebook_entry(tokens)
        elif num_level > 0:
            z_quantized = self.residual_quantize.get_codebook_entry(tokens, num_level=num_level-1)
        
        ss = int(math.sqrt(float(z_quantized.size(1)))) if len(z_quantized.shape) <= 3 else int(z_quantized.size(1))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        return z_quantized


if __name__ == "__main__":
    from omegaconf import OmegaConf
    device = "cuda:1"
    config = OmegaConf.load("maskbit/configs/tokenizer/ft-maskbit-tokenizer-10bit-2lvl.yaml").model.vq_model
    model = FTConvVQModel(config).to(device)
    x = torch.randn(2, 3, 256, 256).requires_grad_().to(device)
    z_quantized, result_dict = model.encode(x)
    print(z_quantized.shape)
    decoded = model.decode(z_quantized)
    print(decoded.shape)
    out, _ = model.forward(x)
    print((decoded-out).abs().mean())
    tokens = result_dict['min_encoding_indices'][1]
    context, _ = model.encode(x, num_levels=1)
    decoded = model.decode_tokens(tokens, num_level=1, context=context)
    print((decoded-out).abs().mean())
    decoded = model.decode_tokens(result_dict['min_encoding_indices'])
    print((decoded-out).abs().mean())