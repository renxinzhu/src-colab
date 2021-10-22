import math
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from util import get_psi_from_state_dict, get_sigma_from_state_dict


class DModule(nn.Module):
    """ Base Class for decomposed layers.

        It contains the methods to get sigma and psi
    """

    def get_sigma_tensors(self):
        """ return sigma, bias not included """
        return get_sigma_from_state_dict(self.state_dict(), False)

    def get_psi_tensors(self):
        """ return psi, bias not included """
        return get_psi_from_state_dict(self.state_dict(), False)

    def get_sigma_parameters(self, bias: bool):
        """ return sigma. bias is incldued if bias=True """
        for name, parameter in self.named_parameters():
            if ('bias' in name if bias else False) or 'sigma' in name:
                yield parameter

    def get_psi_parameters(self, bias: bool):
        """ return psi. bias is incldued if bias=True """
        for name, parameter in self.named_parameters():
            if ('bias' in name if bias else False) or 'psi' in name:
                yield parameter


class DLinear(DModule):
    """ Decomposed linear layers
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.sigma = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        with torch.no_grad():
            psi_factor = 0.2

            self.sigma = Parameter(torch.clone(self.sigma * (1-psi_factor)))
            self.psi = Parameter(torch.clone(self.sigma * psi_factor))

    def reset_parameters(self):
        init.kaiming_uniform_(self.sigma, a=math.sqrt(5))

        if self.bias is not None:
            weight = self.sigma
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.psi + self.sigma
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class D_ConvNd(DModule):
    """ Base class for decomposed convolution layers
    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if transposed:
            self.sigma = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.sigma = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        with torch.no_grad():
            psi_factor = 0.2

            self.sigma = Parameter(torch.clone(self.sigma * (1-psi_factor)))
            self.psi = Parameter(torch.clone(self.sigma * psi_factor))

    def reset_parameters(self):
        init.kaiming_uniform_(self.sigma, a=math.sqrt(5))

        if self.bias is not None:
            weight = self.sigma
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class DConv2d(D_ConvNd):
    """ Decomposed Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.sigma + self.psi
        return self.conv2d_forward(input, weight)
