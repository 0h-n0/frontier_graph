from typing import Tuple, List
import itertools

import torchex.nn as exnn


def conv2d(out_channels, kernel_size, stride):
    return exnn.Conv2d(out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride)


#  https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
def conv2d_output_size(input_size, out_channels, kernel_size, stride, padding=0, dilation=1):
    _, h_in, w_in = input_size
    h_out = int((h_in + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return out_channels, h_out, w_out


def conv_output_size(input_size, kernel_size, stride, padding=0, dilation=1):
    output_size = int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return output_size


def find_conv_layer(input_size, output_size, kernel_sizes: List[int], strides: List[int]) -> Tuple[int, int]:
    for k, s in itertools.product(kernel_sizes, strides):
        if conv_output_size(input_size, k, s) == output_size:
            return k, s
    raise ValueError("not found")
