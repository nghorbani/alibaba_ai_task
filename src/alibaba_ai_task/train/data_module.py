# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
import glob
import os.path as osp
from os import path as osp
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class PriceDataLoader(Dataset):
    """
    This is a dataloder supposed to be used for training Alibaba Price Oracle.
    """

    def __init__(self, dataset_dir: Union[str, Path]):
        assert osp.exists(dataset_dir), FileNotFoundError(
            f'dataset_dir does not exist: {dataset_dir}')
        expected_split_names = ['train', 'vald', 'test']
        assert np.any([dataset_dir.endswith(k) for k in expected_split_names]), \
            ValueError(f'dataset_dir should include one of the expected_split_names: {expected_split_names}')
        self.split_name = split_name = dataset_dir.split('/')[-1]

        ds = {}
        for data_fname in glob.glob(osp.join(dataset_dir, '*.pt')):
            k = osp.basename(data_fname).replace('.pt', '')
            ds[k] = torch.load(data_fname).type(torch.float32)

        n_data = len(ds['price'])

        logger.debug('dimensions of loaded data: {}'.format({k: v.shape for k, v in ds.items()}))

        self.ds = ds

        logger.debug(
            f'Split {split_name}: Loaded #{self.__len__()} data points from dataset_dir {dataset_dir}.')


    def __len__(self):
        k = 'price'
        # k = list(self.ds.keys())[0]
        return len(self.ds[k])

    def __getitem__(self, dIdx):
        return self.fetch_data(dIdx)

    def fetch_data(self, dIdx):
        """
        Dimension of each data_point: time X num_airlines X num_feats
        Parameters
        ----------
        dIdx

        Returns
        -------

        """

        np.random.seed(None)

        data = {k: self.ds[k][dIdx] for k in self.ds.keys()}

        return data


class APODATAModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        logger.debug('Setting up the APO data loader')

        self.data_id = cfg.apo.data_id


        self.batch_size = self.cfg.train_parms.batch_size


        self.example_input_array = {'price': torch.ones(self.batch_size,
                                                         cfg.data_parms.history_length,
                                                         cfg.data_parms.future_length, 1)}

    @rank_zero_only
    def prepare_data(self):

        cfg = self.cfg
        ### the code from te jupyter notebook goes here



    def setup(self, stage: Optional[str] = None):
        # # self.dims is returned when you call dm.size()
        # # Setting default dims here because we know them.
        # # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            for split_name in ['vald', 'train', 'test']:

                dataset = PriceDataLoader(dataset_dir=osp.join(self.cfg.dirs.dataset_dir, split_name))
                assert len(dataset) != 0, ValueError('No data point available!')
                self.__setattr__(f'apo_{split_name}', dataset)
                # self.__setattr__('size'.format(split_name), dataset.shape)

    def train_dataloader(self):
        return DataLoader(self.apo_train,
                          batch_size=self.batch_size,
                          drop_last=False,
                          shuffle=True,
                          num_workers=self.cfg.train_parms.num_workers,
                          persistent_workers=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.apo_vald,
                          batch_size=self.batch_size,
                          drop_last=False,
                          shuffle=False,
                          num_workers=self.cfg.train_parms.num_workers,
                          persistent_workers=True,

                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.apo_test,
                          batch_size=self.batch_size,
                          drop_last=False,
                          shuffle=False,
                          num_workers=self.cfg.train_parms.num_workers,
                          persistent_workers=True,

                          pin_memory=True)


def create_expr_message(cfg):
    expr_msg = '------------------------------------------\n'
    expr_msg += f'[{cfg.apo.expr_id}] batch_size = {cfg.train_parms.batch_size}.\n'
    expr_msg += f'Given a previous {cfg.data_parms.history_length} days price history APO predicts {cfg.data_parms.future_length} price future horizon.\n'
    expr_msg += 'Each data point the time sequence of prices\n'
    expr_msg += f'** Using optimizer.args: {" - ".join([f"{k}: {v}" for k, v in cfg.train_parms.optimizer.args.items()])}\n'

    expr_msg += f'** Using loss weighting: {" - ".join([f"{k}: {v:.2f}" for k, v in cfg.train_parms.loss_weights.items()])}\n'

    expr_msg += '-----------------------------------------\n'

    return expr_msg
