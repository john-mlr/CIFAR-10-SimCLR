import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """SimCLR model.

        Args:
            num_ftrs (int): length of features output by encoder
            encoder (nn.Module): model encoder
    """

    def __init__(self, num_ftrs, encoder):
        super(SimCLR, self).__init__()
        self.num_ftrs = num_ftrs
        self.encoder = encoder
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(nn.Linear(self.num_ftrs, self.num_ftrs, bias=True),
                                nn.ReLU(),
                                nn.Linear(self.num_ftrs, 64, bias=False))

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j


class LinearClassifier(nn.Module):
    """Linear evaluation for SimCLR features

        Args:
            in_ftrs (int): number of features output by the SimCLR encoder
            out_ftrs (int): number of classes
    """

    def __init__(self, in_ftrs, out_ftrs):
        super(LinearClassifier, self).__init__()
        self.in_ftrs= in_ftrs
        self.out_ftrs = out_ftrs

        self.net = nn.Linear(self.in_ftrs, self.out_ftrs)

    def forward(self, x):
        z = self.net(x)
        return z