from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    return (
        input.contiguous().view(batch, channel, new_height, kh, new_width, kw).permute(0, 1, 2, 4, 3, 5),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling over a tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    kh, kw = kernel
    reshaped_input, new_height, new_width = tile(input, kernel)
    
    return reshaped_input.mean(dim=5).view(batch, channel, new_height, new_width, kh).mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax

    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Max reduction over a dimension of a tensor."""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max is argmax."""
        input, dim = ctx.saved_values
        return argmax(input, int(dim.item())) * grad_output, 0.0
        

def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction over a dimension of a tensor."""
    return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling over a tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    kh, kw = kernel
    reshaped_input, new_height, new_width = tile(input, kernel)
    
    x = max(reshaped_input, 5).view(batch, channel, new_height, new_width, kh)
    x = max(x, 4).view(batch, channel, new_height, new_width)
    return x


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropout
        ignore: ignore dropout (for testing)

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > p)
    
    
def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply softmax to a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    return input.exp() / input.exp().sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Apply log softmax to a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply log softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    max_val = max(input, dim)
    return input - max_val - (input - max_val).exp().sum(dim=dim).log()