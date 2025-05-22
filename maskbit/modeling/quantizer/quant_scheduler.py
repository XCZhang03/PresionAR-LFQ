import torch
import numpy as np
from typing import List, Tuple, Union

def agg_quantized(
    quantized_list: List[torch.Tensor],
    num_levels: List,
):
    """
    Aggregate quantized tensors into a single tensor.
    Args:
        quantized_list (List[torch.Tensor]): List of quantized tensors. length: num_quantizers.
        num_levels (List): List of number of levels for each quantizer. 1 <= num_levels <= num_quantizers. length: bs
    Returns:
        torch.Tensor: Aggregated quantized tensor.
    """
    # Initialize the aggregated tensor with zeros
    agg_tensor = torch.zeros_like(quantized_list[0])
    bs = quantized_list[0].shape[0]
    
    # Iterate over each quantized tensor and its corresponding number of levels
    for i in range(len(quantized_list)):
        quantized = quantized_list[i]
        mask = torch.zeros(bs, dtype=torch.bool).to(quantized.device)
        for b in range(bs):
            mask[b] = i < num_levels[b]
        # Add the quantized tensor to the aggregated tensor where the mask is True
        agg_tensor[mask] += quantized[mask]

    return agg_tensor

class QuantScheduler:
    '''
    A scheduler to determine the number of quantizers to use within each batch.
    '''
    def __init__(
        self,
        num_quantizers: int,
        max_train_steps: int = 1_350_000,
        schedule_type: str = 'uniform',
        batch_size: int = 32,
        weights: List[float] = None,
    ):
        self.num_quantizers = num_quantizers
        self.max_train_steps = max_train_steps
        self.schedule_type = schedule_type
        if weights is None:
            weights = [1.0] * num_quantizers
        self.weights = [w / sum(weights) for w in weights]
        assert len(self.weights) == num_quantizers and sum(self.weights) == 1.
        assert self.schedule_type in ['uniform', 'weighted']
        self.batch_size = batch_size
        self.global_step = 0

    def set_step(self, step: int):
        """
        Set the current step of the scheduler.
        Args:
            step (int): The current step.
        """
        self.global_step = step

    def get_num_levels(self) -> List[int]:
        """
        Get the number of levels for each batch.
        Returns:
            List[int]: The number of levels for each batch.
        """
        if self.schedule_type == 'uniform':
            return np.random.randint(1, self.num_quantizers + 1, size=self.batch_size).tolist()
        elif self.schedule_type == 'weighted':
            num_levels = []
            for i in range(self.batch_size):
                num_levels.append(np.random.choice(self.num_quantizers, p=self.weights) + 1)
            return num_levels


if __name__ == "__main__":
    # Example usage
    quant_scheduler = QuantScheduler(num_quantizers=8)
    quant_scheduler.set_step(1000)
    num_levels = quant_scheduler.get_num_levels()
    print(num_levels)