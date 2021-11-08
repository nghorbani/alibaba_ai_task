# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19

import torch
from alibaba_ai_task.models.model_components import Transpose, ResConv1DBlock, SDivide
from alibaba_ai_task.models.transformer import LayeredSelfAttention
from omegaconf import DictConfig
from torch import nn


def masked_mean(tensor, mask, dim, keepdim=False):
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)  # Find the average!


class APO(nn.Module):
    """
    The idea is to process the 7 days history data across the market and across time
    Main component is the self attention unit applied first along the time axis and then along the market
    Market consists of existing detected airlines with sufficient data.
    """

    def __init__(self, cfg: DictConfig):
        super(APO, self).__init__()

        num_attention_feat = 125
        num_attention_layers = cfg.model_parms.num_attention_layers
        num_attention_heads = 5

        self.pred_time_length = cfg.data_parms.history_length + cfg.data_parms.future_length

        # IN (N X history X  num_airlines X 1)
        # Reshape/Transpose
        # IN ((N * num_airlines) X history X  1)
        self.price_features = nn.Sequential(
            Transpose(-2, -1),
            nn.BatchNorm1d(1),
            # ResConv1DBlock(1, num_attention_feat*2, enable_bn = True),
            ResConv1DBlock(1, num_attention_feat, enable_bn=True),
            nn.LeakyReLU(0.2),
            # ResConv1DBlock( num_attention_feat * 2, num_attention_feat, enable_bn = True),
            # nn.LeakyReLU(0.2),
        )
        # OUT ((N * num_airlines) X num_feats X time)

        # IN (N*num_airlines X history X  num_feats)
        self.price_dynamics_attention = LayeredSelfAttention(num_attention_feat,
                                                             num_attention_layers,
                                                             num_attention_heads)
        # OUT ((N * num_airlines) X num_feats X time)

        # IN ((N * num_airlines) X num_feats X time)
        self.price_prediction = nn.Sequential(nn.LeakyReLU(0.2),  # input N*
                                              ResConv1DBlock(num_attention_feat, num_attention_feat, enable_bn=True),
                                              nn.LeakyReLU(0.2),
                                              SDivide(num_attention_feat ** .5),
                                              Transpose(-2, -1),  # OUT ((N * num_airlines) X history X  num_feats)
                                              ResConv1DBlock(cfg.data_parms.history_length, self.pred_time_length,
                                                             enable_bn=True),
                                              nn.LeakyReLU(0.2),
                                              # OUT ((N * num_airlines) X (time+future_time) X num_feats)
                                              Transpose(-2, -1),
                                              )
        # OUT ((N * num_airlines) X num_feats X (time+future_time))
        # Reshape/Transpose

        # IN ((N * (time+future_time)) X num_feats X num_airlines)
        self.market_attention = LayeredSelfAttention(num_attention_feat,
                                                     num_attention_layers,
                                                     num_attention_heads)
        # OUT ((N * (time+future_time)) X num_feats X num_airlines)

        self.market_feat = nn.Sequential(
            nn.LeakyReLU(0.2),
            SDivide(num_attention_feat ** .5),
            ResConv1DBlock(num_attention_feat, 1, enable_bn=False),
            nn.ReLU(),  # ((N * (time+future_time)) X 1 X num_airlines)
            Transpose(-2, -1),
        )
        # OUT ((N * (time+future_time)) X num_airlines X 1)

    def forward(self, price):
        """

        Args:
            price: bs x time_length x num_airlines x num_feats
            feat 1 is the availability mask

        Returns:

        """
        # Todo: does it make sense to normalize the input data
        bs, time_length, num_airlines, num_feats = price.shape

        availability_mask = price[:, :, :, 1:]
        price = price[:, :, :, :1]

        price_std, price_mean = torch.std_mean(price, dim=[1, 3], keepdim=True, unbiased=True)  # mean/std over time
        price_norm = ((price - price_mean) / price_std)
        price_norm = torch.nan_to_num(price_norm, nan=0)

        airline_flattened_price = price_norm.transpose(1, 2).contiguous().view(bs * num_airlines, time_length, 1)

        price_feats = self.price_features(airline_flattened_price)
        price_dynamics_att, price_dynamics_att_wts = self.price_dynamics_attention(price_feats)
        price_dynamics_pred = self.price_prediction(price_dynamics_att)
        price_dynamics_pred = price_dynamics_pred.view(bs, num_airlines, -1, self.pred_time_length).transpose(1, -1)
        price_time_flattened = price_dynamics_pred.contiguous().view(bs * self.pred_time_length, -1, num_airlines)
        price_market_att, price_market_att_wts = self.market_attention(price_time_flattened)
        market_pred = self.market_feat(price_market_att).view(bs, self.pred_time_length, num_airlines, 1)

        market_pred = ((market_pred * price_std) + market_pred)  # *availability_mask
        market_pred = torch.nan_to_num(market_pred, nan=0)

        output_dict = {
            'price': market_pred,
            'price_dynamics_att_wts': price_dynamics_att_wts,
            'price_market_att_wts':price_market_att_wts
            }

        return output_dict