"""Radio backgrdound emulator model."""

import torch
from torch import nn


def modulelist2sequential(module: nn.ModuleList) -> nn.Sequential:
    """Convert a nn.ModuleList to a nn.Sequential.

    Parameters
    ----------
    module : nn.ModuleList
        The module to convert.

    Returns
    -------
    nn.Sequential
        The converted module.
    """
    out = nn.Sequential()
    for layer in module:
        out.append(layer)
    return out


class CNN(nn.Module):
    """A simple CNN model.

    Parameters
    ----------
    nconvs : int
        Number of convolutional layers.
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    hid_ch : int, optional
        Number of channels in the hidden layer convolutions (when nconvs > 2).
    kernel_size : int, optional
        Kernel size for the convolutional layers.
    stride : int, optional
        Stride for the convolutional layers.
    padding : int, optional
        Padding for the convolutional layers.
    dropout : bool, optional
        Whether to apply dropout.
    f_dropout : float, optional
        Dropout rate. Default is 0.1.
    final_act : bool, optional
        Whether to apply an activation function to the final layer.
    batch_norm : bool, optional
        Whether to apply batch normalization.
    act_fn : object, optional
        Activation function to apply. Default is nn.LeakyReLU.

    """

    def __init__(
        self,
        nconvs: int,
        in_ch: int,
        out_ch: int,
        hid_ch: int = None,
        kernel_size: tuple = (2,),
        stride: tuple = (1,),
        padding: tuple = (0,),
        dropout: bool = False,
        f_dropout: float = 0.1,
        final_act: bool = False,
        batch_norm: bool = False,
        act_fn: object = nn.LeakyReLU,
    ):
        super().__init__()
        self.cnn = cnn_list(
            nconvs=nconvs,
            in_ch=in_ch,
            out_ch=out_ch,
            hid_ch=hid_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            final_act=final_act,
            batch_norm=batch_norm,
            act_fn=act_fn,
        )

    def forward(self, x):
        """Forward pass of the model."""
        y = self.cnn(x)
        return y


def cnn_list(
    nconvs: int,
    in_ch: int,
    out_ch: int,
    hid_ch: int = None,
    kernel_size: tuple = (2,),
    stride: tuple = (1,),
    padding: tuple = (0,),
    dropout: bool = False,
    f_dropout: float = 0.1,
    final_act: bool = False,
    batch_norm: bool = False,
    act_fn: object = nn.LeakyReLU,
):
    """Create a nn.ModuleList of convolutional layers.

    Parameters
    ----------
    nconvs : int
        Number of convolutional layers.
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    hid_ch : int, optional
        Number of channels in the hidden layer convolutions (when nconvs > 2).
    kernel_size : int, optional
        Kernel size for the convolutional layers.
    stride : int, optional
        Stride for the convolutional layers.
    padding : int, optional
        Padding for the convolutional layers.
    dropout : bool, optional
        Whether to apply dropout.
    f_dropout : float, optional
        Dropout rate. Default is 0.1.
    final_act : bool, optional
        Whether to apply an activation function to the final layer.
    batch_norm : bool, optional
        Whether to apply batch normalization.
    act_fn : object, optional
        Activation function to apply. Default is nn.LeakyReLU.

    Returns
    -------
    nn.ModuleList
        The list of convolutional layers
    """
    if hid_ch is None:
        hid_ch = out_ch
    conv_in = nn.Sequential(
        nn.ConvTranspose2d(
            in_ch, hid_ch, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        act_fn(),
    )
    conv_hid = nn.Sequential(
        nn.ConvTranspose2d(
            hid_ch, hid_ch, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        act_fn(),
    )
    if batch_norm:
        conv_in.append(nn.BatchNorm2d(hid_ch))
        conv_hid.append(nn.BatchNorm2d(hid_ch))
    if dropout:
        conv_in.append(nn.Dropout(f_dropout))
        conv_hid.append(nn.Dropout(f_dropout))
    cnn = modulelist2sequential(nn.ModuleList([conv_hid for i in range(nconvs - 2)]))
    if final_act:
        conv_out = nn.Sequential(
            nn.ConvTranspose2d(
                hid_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            act_fn(),
        )
    else:
        conv_out = nn.ConvTranspose2d(
            hid_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
        )
    return conv_in.extend(cnn.append(conv_out))


class Radio_Emulator(nn.Module):
    """Neural network model for the radio background emulator."""

    def __init__(
        self,
        nlayers: tuple = (10, 10, 5, 5, 3),
        nnodes: tuple = (1500, 1000, 500, 64 * 20, 300),
        out_len: tuple = (103, 103, 103, 25 * 20, 1),
        input_len: int = 5,
        ps_inp_shape: tuple = (64, 5, 4),
    ):
        super().__init__()
        self.ps_inp_shape = ps_inp_shape
        inp_nnodes = input_len
        Tb_block = nn.Sequential(nn.Linear(nnodes[0], nnodes[0]), nn.LeakyReLU())
        Tb_branch = nn.Sequential(
            nn.Linear(inp_nnodes, nnodes[0]),
            nn.LeakyReLU(),
        )
        Tb_branch.append(
            modulelist2sequential(
                nn.ModuleList([Tb_block for i in range(nlayers[0] - 2)])
            )
        )
        Tb_branch.append(nn.Sequential(nn.Linear(nnodes[0], out_len[0])))
        self.Tb_branch = Tb_branch

        Tr_block = nn.Sequential(nn.Linear(nnodes[1], nnodes[1]), nn.LeakyReLU())
        Tr_branch = nn.Sequential(
            nn.Linear(inp_nnodes, nnodes[1]),
            nn.LeakyReLU(),
        )
        Tr_branch.append(
            modulelist2sequential(
                nn.ModuleList([Tr_block for i in range(nlayers[1] - 2)])
            )
        )
        Tr_branch.append(nn.Sequential(nn.Linear(nnodes[1], out_len[1])))
        self.Tr_branch = Tr_branch

        xHI_block = nn.Sequential(nn.Linear(nnodes[2], nnodes[2]), nn.LeakyReLU())
        xHI_branch = nn.Sequential(
            nn.Linear(inp_nnodes, nnodes[2]),
            nn.LeakyReLU(),
        )
        xHI_branch.append(
            modulelist2sequential(
                nn.ModuleList([xHI_block for i in range(nlayers[2] - 2)])
            )
        )
        xHI_branch.append(nn.Sequential(nn.Linear(nnodes[2], out_len[2]), nn.Sigmoid()))
        self.xHI_branch = xHI_branch

        ps_branch = nn.Sequential(
            nn.Linear(inp_nnodes, nnodes[3]),
            nn.LeakyReLU(),
        )
        ps_block = nn.Sequential(nn.Linear(nnodes[3], nnodes[3]), nn.LeakyReLU())
        ps_branch.append(
            modulelist2sequential(
                nn.ModuleList([ps_block for i in range(nlayers[3] - 1)])
            )
        )
        self.ps_fc = ps_branch

        self.cnn1 = CNN(
            nconvs=2,
            in_ch=64,
            out_ch=64,
            hid_ch=64,
            kernel_size=(2, 2),
            batch_norm=False,
            final_act=True,
        )
        self.cnn2 = CNN(
            nconvs=2,
            in_ch=64,
            out_ch=64,
            hid_ch=64,
            kernel_size=(3, 2),
            batch_norm=False,
            final_act=True,
        )
        self.cnn2v2 = CNN(
            nconvs=2,
            in_ch=64,
            out_ch=32,
            hid_ch=32,
            kernel_size=(3, 3),
            batch_norm=False,
            final_act=True,
        )
        self.cnn3 = CNN(
            nconvs=2,
            in_ch=32,
            out_ch=1,
            hid_ch=16,
            kernel_size=(3, 3),
            batch_norm=False,
            final_act=True,
        )

        tau_block = nn.Sequential(nn.Linear(nnodes[-1], nnodes[-1]), nn.LeakyReLU())
        tau_branch = nn.Sequential(
            nn.Linear(inp_nnodes, nnodes[-1]),
            nn.LeakyReLU(),
        )
        tau_branch.append(
            modulelist2sequential(
                nn.ModuleList([tau_block for i in range(nlayers[-1] - 2)])
            )
        )
        tau_branch.append(nn.Sequential(nn.Linear(nnodes[-1], out_len[-1])))
        self.tau_branch = tau_branch

    def init_weights(self, m):
        """Initialize the weights of the model."""
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            # So that xHI starts around 0
            torch.nn.init.uniform_(m.bias, -4.0, -1.0)

    def forward(self, input_params):
        """Forward pass of the model."""
        Tb_pred = self.Tb_branch(input_params)
        Tr_pred = self.Tr_branch(input_params)
        xHI_pred = self.xHI_branch(input_params)
        tau_pred = self.tau_branch(input_params)
        ps_1d = self.ps_fc(input_params)
        ps_2d = torch.reshape(ps_1d, (ps_1d.shape[0],) + tuple(self.ps_inp_shape))
        ps = self.cnn1(ps_2d)
        ps = self.cnn1(ps)
        ps = self.cnn2(ps)
        ps = self.cnn2(ps)
        ps = self.cnn2v2(ps)
        ps = self.cnn3(ps)
        ps_pred = torch.reshape(ps, (ps.shape[0], ps.shape[-2] * ps.shape[-1]))

        return torch.cat((Tb_pred, Tr_pred, xHI_pred, ps_pred, tau_pred), 1)
