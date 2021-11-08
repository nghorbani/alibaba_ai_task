# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
from torch import nn

class ResConv1DBlock(nn.Module):
    """ a series of 1D convolutions with residuals"""

    def __init__(self, num_feat_in, num_feat_out, num_h=256, enable_bn=True):
        super(ResConv1DBlock, self).__init__()
        res_conv1_components = [
            nn.Conv1d(num_feat_in, num_h, 1, 1),
            nn.BatchNorm1d(num_h),
            nn.LeakyReLU(0.2),
            nn.Conv1d(num_h, num_feat_out, 1, 1),
            nn.BatchNorm1d(num_feat_out)
        ]

        self.res_conv1d = nn.Sequential(*([el for el in res_conv1_components
                                           if not isinstance(el, nn.BatchNorm1d) or enable_bn]))

        res_conv1d_short = [nn.Conv1d(num_feat_in, num_feat_out, 1, 1)]
        if enable_bn:
            res_conv1d_short += [nn.BatchNorm1d(num_feat_out)]
        self.res_conv1d_short = nn.Sequential(*(res_conv1d_short if num_feat_in != num_feat_out else [nn.Identity()]))

    def forward(self, x):
        return self.res_conv1d(x) + self.res_conv1d_short(x)


class LayeredResConv1d(nn.Module):

    def __init__(self, num_feat_in, num_feat_out, num_layers, num_h=256, enable_bn = True ):
        super(LayeredResConv1d, self).__init__()

        block_blue_print = [el for el in [ResConv1DBlock(num_feat_in, num_feat_out * 3),
                          nn.LeakyReLU(0.2),
                          nn.Conv1d(num_feat_out * 3, num_feat_out, kernel_size=1),
                          nn.BatchNorm1d(num_feat_out),
                          nn.LeakyReLU(0.2),
                          nn.Conv1d(num_feat_out, num_feat_out, kernel_size=1)]
                                           if not isinstance(el, nn.BatchNorm1d) or enable_bn]
        self.res_conv1d_blocks = nn.ModuleList([nn.Sequential(*block_blue_print) for _ in range(num_layers)])

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        for res_conv1d_block in self.res_conv1d_blocks:
            new_point_feats = res_conv1d_block(point_feats)
            point_feats = point_feats + new_point_feats

        return point_feats

class Contiguous(nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()
        self._name = 'contiguous'

    def forward(self, x):
        return x.contiguous()


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args
        self._name = 'permute'

    def forward(self, x):
        return x.permute(self.shape)


class Transpose(nn.Module):
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.shape = args
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.shape)


class SDivide(nn.Module):
    def __init__(self, scale):
        super(SDivide, self).__init__()
        self.scale = scale
        self._name = 'scalar_divide'

    def forward(self, x):
        return x / self.scale


class SelectItem(nn.Module):
    # https://stackoverflow.com/a/54660829
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
