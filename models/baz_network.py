import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
BAZ network.

Reference:
    [1] S. M. Mousavi and G. C. Beroza. (2020) 
        Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations.
        IEEE Transactions on Geoscience and Remote Sensing, 58, 11, 8211-8224.
        doi: 10.1109/TGRS.2020.2988770.
"""


class BAZ_Network(nn.Module):
    """
    BAZ network
    """

    def __init__(
        self,
        in_channels: int = 3,
        in_samples: int = 8192,
        in_matrix_dim: int = 7,
        conv_channels: list = [20, 32, 64, 20],
        kernel_size: int = 3,
        pool_size: int = 2,
        lin_hidden_dim: int = 100,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        dim = in_samples
        for inc, outc in zip([in_channels] + conv_channels[:-1], conv_channels):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=inc,
                        out_channels=outc,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.MaxPool1d(pool_size, ceil_mode=True),
                )
            )
            dim = (dim + (pool_size - (dim % pool_size)) % pool_size) // pool_size
        dim = (dim + in_matrix_dim) * conv_channels[-1]

        self.flatten0 = nn.Flatten()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=conv_channels[-1], kernel_size=1
        )
        self.relu0 = nn.ReLU()
        self.flatten1 = nn.Flatten()

        self.lin0 = nn.Linear(in_features=dim, out_features=lin_hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.lin1 = nn.Linear(in_features=lin_hidden_dim, out_features=2)

    @torch.no_grad()
    def _cov(self, x: torch.Tensor):
        """
        `torch.cov` does not support batch computing, so the function was reimplemented.
        """
        N, C, L = x.size()
        diff = (x - x.mean(-1, keepdim=True)).transpose(-1, -2).reshape(N * L, C)
        cov = torch.bmm(diff.unsqueeze(-1), diff.unsqueeze(-2)).view(N, L, C, C).sum(
            dim=1
        ) / (L - 1)
        return cov

    @torch.no_grad()
    def _eig(self, cov: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Computes eigenvalues and eigenvectors. Returns only the real part of the complex numbers.
        """
        eig_values, eig_vectors = torch.linalg.eig(cov)
        eig_values, eig_vectors = eig_values.unsqueeze(-1).type(dtype), eig_vectors.type(dtype)
        return eig_values, eig_vectors

    @torch.no_grad()
    def _compute_cov_and_eig(self, x):
        """
        Computes covariance matrix, eigenvalues and the eigenvectors.
        """
        cov_mat = self._cov(x)

        eig_values, eig_vectors = self._eig(cov_mat, x.dtype)

        eig_values /= eig_values.max()
        cov_mat /= cov_mat.abs().max()
        x = torch.cat([cov_mat, eig_values, eig_vectors], dim=-1)

        return x

    def forward(self, x):
        x1 = self._compute_cov_and_eig(x)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten0(x)

        x1 = self.conv1(x1)
        x1 = self.relu0(x1)
        x1 = self.flatten1(x1)

        x = torch.cat([x, x1], dim=1)
        x = self.lin0(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.lin1(x)

        return x[:, :1], x[:, 1:]


@register_model
def baz_network(**kwargs):
    model = BAZ_Network(**kwargs)
    return model
