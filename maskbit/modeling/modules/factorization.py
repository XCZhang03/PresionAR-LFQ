"""This file contains the definition of utility functions to group tokens."""

import math
import torch
from typing import Tuple

def get_codebook_config(codebook_size: int=None, bits: int=None, variants: int=None) -> Tuple[int, int, int]:
    if variants is None:
        variants = 2
    if bits is None:
        assert codebook_size is not None, "Either bits or codebook_size must be provided."
        bits = int(math.log(codebook_size, variants))
    
    if codebook_size is None:
        codebook_size = variants ** bits
    
    assert codebook_size == variants ** bits, \
        f"codebook_size {codebook_size} must be equal to variants ** bits {variants} ** {bits}"
    
    return codebook_size, bits, variants


def combine_factorized_tokens(tokens: torch.Tensor, codebook_size: int=None, splits: int=1, bits: int=None, variants: int=None) -> torch.Tensor:
    """
    Combine the tokens into a single token.

    Args:
        tokens -> torch.Tensor: Tensor of shape (batch_size, n, m).
        codebook_size -> int: The size of the codebook.
        splits -> int: Number of splits.
    
    Returns:
        combined_tokens -> torch.Tensor: Tensor of shape (batch_size, n).
    """
    combined_tokens = torch.zeros((tokens.shape[0], tokens.shape[1]), device=tokens.device)
    codebook_size, bits, variants = get_codebook_config(codebook_size, bits, variants)
    bit_shift = bits // splits 
    for i in range(splits):
        combined_tokens += (tokens[..., i] * (variants ** (i * bit_shift))).long()

    return combined_tokens


def split_factorized_tokens(tokens: torch.Tensor, codebook_size: int=None, splits: int=None, bits: int=None, variants: int=None) -> torch.Tensor:
    """
    Split the tokens into multiple tokens.

    Args:
        tokens -> torch.Tensor: Tensor of shape (batch_size, n).
        codebook_size -> int: The size of the codebook.
        splits -> int: Number of splits.
    
    Returns:
        split_tokens -> torch.Tensor: Tensor of shape (batch_size, n, m).
    """
    codebook_size, bits, variants = get_codebook_config(codebook_size, bits, variants)
    bit_shift = bits // splits 
    
    basis = variants ** bit_shift

    split_tokens = []
    for i in range(splits):
        split_tokens.append((tokens // (basis ** i)) % basis)

    return torch.stack(split_tokens, dim=-1)


if __name__ == "__main__":
    tokens = torch.randint(0, 59048, (1, 16))
    split_tokens = split_factorized_tokens(tokens, None, 1, 10, 3)

    assert split_tokens.shape == (1, 16, 1)
    assert split_tokens.dtype == torch.int64

    combined_tokens = combine_factorized_tokens(split_tokens, None, 1, 10, 3)

    assert (tokens == combined_tokens).all()

    split_tokens = split_factorized_tokens(tokens, None, 2, 10, 3)
    combined_tokens = combine_factorized_tokens(split_tokens, None, 2, 10, 3)

    assert split_tokens.shape == (1, 16, 2)
    assert (tokens == combined_tokens).all(), f"{tokens} != {combined_tokens}"

    # assert (torch.bitwise_right_shift(tokens, 5) == split_tokens[..., 1]).all()
    assert (tokens % 243  == split_tokens[..., 0]).all()