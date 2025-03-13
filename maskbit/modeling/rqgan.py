"""This file contains the definition of the our tokenizer, which can use VQ or LFQ."""

import math
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange

from modeling.modules import (BaseModel, ConvDecoder, ConvDecoderLegacy,
                              ConvEncoder)
# from modeling.quantizer import LookupFreeQuantizer, SimpleVectorizer
from modeling.quantizer.residual_lfq import ResidualLFQ


def choose_vector_quantizer_class(config):
    if config.quantizer_type == "residual_lfq":
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

class RQModel(BaseModel):
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

        self.codebook_size = [q.codebook_size for q in self.quantize.quantizers]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor, num_levels: int=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Encodes the input tensor, i.e. runs the encoder.

        Args:
            x -> torch.Tensor: The input tensor.
            num_levels -> int: The number of levels to quantize to.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = self.encoder(x)
        z_quantized, result_dict = self.quantize(z, num_levels=num_levels)
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
            tokens -> torch.Tensor: The token indices. shape: (n, b, h, w)

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        z_quantized = self.quantize.get_codebook_entry(tokens)
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    def forward(self, input: torch.Tensor, num_levels: int=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the model on the input tensor.

        Args:
            input -> torch.Tensor: The input image.
            num_levels -> int: The number of levels to quantize to.

        Returns:
            decoded -> torch.Tensor: The decoded image.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        if self.finetune_decoder:
            self.encoder.eval()
            z_quantized, result_dict = self._finetuning_encoder_forward(input,num_levels=num_levels)
        else:
            z_quantized, result_dict = self.encode(input,num_levels=num_levels)

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
    


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("maskbit/configs/tokenizer/rqbit_tokenizer_10bit.yaml").model.vq_model
    model = RQModel(config)
    print(model)
    print(model.codebook_size)
    image = torch.randn(2, 3, 256, 256)
    z_quantized, result_dict = model.encode(image,num_levels=1)
    print(z_quantized.shape)
    decoded = model.decode(z_quantized)
    print(decoded.shape)
    decoded, result_dict = model(image,num_levels=1)
    print(decoded.shape)
