"""This file contains the definition of the look-free quantizer with multivariante codebook."""

from typing import Mapping, Text, Tuple

import torch
from einops import rearrange, reduce

from modeling.quantizer.quantizer_utils import entropy_loss_fn



class MultivariantLFQ(torch.nn.Module):
    def __init__(
        self,
        token_size: int = 10,
        variants: int = 3,
        scale: float = 1.0,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
    ):
        """ Initializes the lookup-free quantizer.

        Args:
            variants -> int: The number of variants for the codebook entries, 2 for binary (-1,1).
            token_size -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
        """
        super().__init__()
        self.token_size = token_size
        self.levels = variants
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        self.codebook_size = variants ** token_size

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        basis = torch.pow(self.levels, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('basis', basis.int(), persistent=False)
        
        self.shift = self.levels // 2 if self.levels % 2 == 1 else self.levels - 1
        self.mult = 2 if self.levels % 2 == 0 else 1
        
        codebook = self.convert_indices_to_tokens(torch.arange(self.codebook_size))
        self.register_buffer('codebook', codebook, persistent=False)
        
    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """ Normalizes the input tensor.

        Args:
            z -> torch.FloatTensor: The input tensor. shape: (batch_size, height, width, channels) range: [0, self.levels - 1]

        Returns:
            z_normalized -> torch.Tensor: The normalized input tensor. range: [- self.shift, self.shift]
        """
        return z * self.mult - self.shift
    
    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """ Denormalizes the input tensor.

        Args:
            z -> torch.FloatTensor: The input tensor. shape: (batch_size, height, width, channels) range: [- self.shift, self.shift]

        Returns:
            z_denormalized -> torch.Tensor: The denormalized input tensor. range: [0, self.levels - 1]
        """
        return (z + self.shift) / self.mult


    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """ Quantizes the input tensor.

        Args:
            z -> torch.FloatTensor: The input tensor. shape: (batch_size, height, width, channels)

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
        """
        bits = (z + self.shift) / self.mult
        bits_quantized = torch.round(bits).clamp(0, self.levels - 1)
        z_quantized = bits_quantized * self.mult - self.shift
        return z_quantized


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Forward pass of the quantizer.

        Args:
            z -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_scaled = z / self.scale
        z_scaled_quantized = self.quantize(z_scaled) 
        min_encoding_indices = self.convert_tokens_to_indices(z_scaled_quantized)
        z_quantized = z_scaled_quantized * self.scale

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            d = - 2 * torch.einsum('b h w c, n c -> b h w n', z, self.codebook * self.scale)

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict

    
    def convert_tokens_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Converts the given tokens to index numbers.

        As the codebook exists only implicitly, this is mainly an integer conversion from a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            tokens -> torch.Tensor: The tokens. normalized to be zero-centered. unscaled such that has interger values

        Returns:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.
        """
        assert tokens.shape[-1] == self.token_size
        bits = self.denormalize(tokens)
        indices = reduce(bits * self.basis, '... c -> ...', 'sum').to(torch.int32)
        return indices

    def convert_indices_to_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """ Converts the given indices to tokens.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation. normalized to be zero-centered. unscaled to have integer values
        """
        indices = indices.long()
        indices = rearrange(indices, '... -> ... 1')
        bits = (indices // self.basis) % self.levels
        tokens = self.normalize(bits)
        return tokens
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the `codebook entry` for the given indices.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        tokens = self.convert_indices_to_tokens(indices)
        tokens = tokens * self.scale
        return tokens




if  __name__ == "__main__":
    quantizer = MultivariantLFQ(
        token_size=10,
        commitment_cost=1.0,
        entropy_loss_weight=0.1,
        entropy_loss_temperature=0.01,
        levels=3,
        scale=0.5,
        entropy_gamma=1.0
    )
    all_entries = torch.arange(1024).reshape(1, 1, 1024)
    indices = quantizer.convert_tokens_to_indices(quantizer.convert_indices_to_tokens(all_entries))
    assert torch.equal(
        indices,
        all_entries
    )
    assert torch.equal(
        quantizer.convert_tokens_to_indices(quantizer.codebook[:1024,...].reshape(1,1,1024,10)),
        all_entries
    )
    image = torch.randn(1, 10, 32, 32)
    quantized_image, result_dict = quantizer(image)
    indices = result_dict['min_encoding_indices']
    reconstructed_image = quantizer.get_codebook_entry(indices).permute(0, 3, 1, 2)  # (b, c, h, w)
    commitment_loss = result_dict['commitment_loss']
    reconstruction_error = torch.mean((image - reconstructed_image) ** 2)
    # assert torch.isclose(
    #     commitment_loss,
    #     reconstruction_error
    # )
