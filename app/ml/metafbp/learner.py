"""Learner architecture for MetaFBP (placeholder for production code from MetaVisionLab/MetaFBP).

This file should be replaced with the actual learner.py from the MetaFBP training repository.
It defines the network architecture using nn.ParameterList for MAML compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Learner(nn.Module):
    """Network architecture for MetaFBP using ParameterList for MAML compatibility.

    The Learner stores all parameters in nn.ParameterList objects:
    - self.vars: trainable parameters (conv weights, fc weights, etc.)
    - self.vars_bn: batch norm running statistics

    Checkpoint keys follow patterns: vars.0, vars.1, ..., vars_bn.0, vars_bn.1, ...

    Architecture: ResNet-18 based feature extractor producing 512-dim embeddings.
    """

    def __init__(self, config=None):
        super(Learner, self).__init__()
        self.config = config or self._default_config()
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        self._build_network()

    def _default_config(self):
        """Default ResNet-18 style configuration."""
        return [
            ('conv2d', [64, 3, 7, 7, 2, 3]),      # conv1
            ('bn', [64]),
            ('relu', [True]),
            ('max_pool2d', [3, 2, 1]),
            # Simplified ResNet blocks
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('conv2d', [128, 64, 3, 3, 2, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('conv2d', [128, 128, 3, 3, 1, 1]),
            ('bn', [128]),
            ('conv2d', [256, 128, 3, 3, 2, 1]),
            ('bn', [256]),
            ('relu', [True]),
            ('conv2d', [256, 256, 3, 3, 1, 1]),
            ('bn', [256]),
            ('conv2d', [512, 256, 3, 3, 2, 1]),
            ('bn', [512]),
            ('relu', [True]),
            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('bn', [512]),
            ('adaptive_avg_pool2d', [1, 1]),
            ('flatten', []),
        ]

    def _build_network(self):
        """Build network parameters from config."""
        for name, param in self.config:
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(*param[:2], *param[2:4]))
                nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, vars=None, bn_training=True):
        """Forward pass through the learner network.

        Args:
            x: Input tensor
            vars: Optional parameter override for MAML inner loop
            bn_training: Whether batch norm is in training mode

        Returns:
            512-dimensional feature embedding
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'adaptive_avg_pool2d':
                x = F.adaptive_avg_pool2d(x, param)
            elif name == 'flatten':
                x = x.view(x.size(0), -1)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

        return x

    def parameters(self):
        return self.vars
