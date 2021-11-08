# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
from alibaba_ai_task.tools.omni_tools import copy2cpu as c2c
from alibaba_ai_task.tools.omni_tools import get_support_data_dir
from alibaba_ai_task.tools.omni_tools import makepath
from alibaba_ai_task.tools.omni_tools import trainable_params_count
from loguru import logger

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module

from alibaba_ai_task.models.apo_model import APO

class APOTrainer(LightningModule):
    """

    """

    def __init__(self, cfg: DictConfig):
        super(APOTrainer, self).__init__()

        self.expr_id = cfg.apo.expr_id
        self.data_id = cfg.apo.data_id

        _support_data_dir = get_support_data_dir(__file__)

        self.work_dir = cfg.dirs.work_dir

        self.train_start_time = dt.now().replace(microsecond=0)

        self.model = APO(cfg)

        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.cfg_fname = osp.join(self.work_dir, f'{self.expr_id}_{self.data_id}.yaml')
        OmegaConf.save(config=self.cfg, f=self.cfg_fname)

    def forward(self, points):
        return self.model(points)

    @rank_zero_only
    def on_train_start(self):
        # if self.global_rank != 0: return
        # shutil.copy2(__file__, self.work_dir)

        ######## make a backup of the repo
        git_repo_dir = os.path.abspath(__file__).split('/')
        git_repo_dir = '/'.join(git_repo_dir[:git_repo_dir.index('alibaba_ai_task') + 1])
        start_time = dt.strftime(self.train_start_time, '%Y_%m_%d_%H_%M_%S')
        archive_path = makepath(self.work_dir, 'code', f'apo_{start_time}.tar.gz', isfile=True)
        os.system(f"cd {git_repo_dir} && git ls-files -z | xargs -0 tar -czf {archive_path}")
        ########
        logger.info(f'Created a git archive backup at {archive_path}')

    def configure_optimizers(self):

        lr_scheduler_class = getattr(lr_sched_module, self.cfg.train_parms.lr_scheduler.type)
        schedulers = []
        optimizers = []

        model_params = [a[1] for a in self.model.named_parameters() if a[1].requires_grad]
        optimizer_class = getattr(optim_module, self.cfg.train_parms.optimizer.type)
        optimizer = optimizer_class(model_params, **self.cfg.train_parms.optimizer.args)
        optimizers.append(optimizer)
        gen_lr_scheduler = lr_scheduler_class(optimizer, **self.cfg.train_parms.lr_scheduler.args)
        schedulers.append({
            'scheduler': gen_lr_scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        })

        logger.info(f'Total trainable model_params: {trainable_params_count(model_params) * 1e-6:2.4f} M.')

        return optimizers, schedulers

    def _compute_loss(self, price_pred, price_gt, availability_mask):
        # criteria = torch.nn.MSELoss(reduction='none')
        criteria = torch.nn.L1Loss(reduction='none')
        # breakpoint()
        loss = torch.mean(criteria(price_pred, price_gt))
        # print(price_pred[0,:2, 0].view(-1), price_gt[0,:2, 0].view(-1))
        # loss = torch.mean(criteria(price_pred, price_gt[:,:self.cfg.data_parms.history_length]))
        # loss = torch.mean(criteria(price_pred, price_gt[:,:self.cfg.data_parms.history_length]) * availability_mask[:,:self.cfg.data_parms.history_length])
        # loss = torch.mean(l2(price_pred, price_gt) * availability_mask)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        cfg = self.cfg

        wts = cfg.train_parms.loss_weights

        drec = self.forward(batch['price'][:,:cfg.data_parms.history_length])

        if optimizer_idx == 0 or optimizer_idx is None:

            price_loss = self._compute_loss(drec['price'], batch['price'][:,:,:,:1], batch['price'][:,:,:,1:])

            # self.log('train_label_loss', label_assignment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            train_loss = wts.price * price_loss

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):

        cfg = self.cfg

        drec = self.forward(batch['price'][:,:cfg.data_parms.history_length])

        price_loss = self._compute_loss(drec['price'], batch['price'][:, :, :, :1], batch['price'][:, :, :, 1:])

        result = {'val_loss': c2c(price_loss).reshape(1)}

        return result

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)

        return {'test_loss': loss['val_loss']}

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, mode='test')

    def validation_epoch_end(self, outputs, mode = 'val'):
        # if self.global_rank != 0: return

        data = {}
        for one in outputs:
            for k, v in one.items():
                if k == 'progress_bar': continue
                if not k in data: data[k] = []
                data[k].append(v)
        # data = {k: np.concatenate(v) for k, v in data.items()}


        metrics = {f'{mode}_loss': np.nanmean(np.concatenate([v[f'{mode}_loss'] for v in outputs]))}

        # if self.global_rank == 0:
            # logger.success(
            #     f'Epoch {self.current_epoch}: {", ".join(f"{k}:{v:.2f}" for k, v in metrics.items())}')
            # logger.info(
            #     f'lr is {["{:.2e}".format(pg["lr"]) for opt in self.trainer.optimizers for pg in opt.param_groups]}')

        metrics = {k if k.startswith(f'{mode}_') else f'{mode}_{k}': torch.tensor(v, device=self.device) for k, v in
                   metrics.items()}

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @rank_zero_only
    def on_train_end(self):

        self.train_endtime = dt.now().replace(microsecond=0)
        endtime = dt.strftime(self.train_endtime, '%Y_%m_%d_%H_%M_%S')
        elapsedtime = self.train_endtime - self.train_start_time

        best_model_basename = self.trainer.checkpoint_callback.best_model_path
        logger.success(f'best_model_fname: {best_model_basename}')
        OmegaConf.save(config=self.cfg, f=self.cfg_fname)

        logger.success(f'Epoch {self.current_epoch} - Finished training at {endtime} after {elapsedtime}')

    @staticmethod
    def prepare_train_cfg(**kwargs) -> DictConfig:

        app_support_dir = get_support_data_dir(__file__)
        base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/apo_train_conf.yaml'))

        override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

        return OmegaConf.merge(base_cfg, override_cfg)


def train_apo_once(job_args):
    """
    This function must be self sufficient with imports to be able to run on the cluster
    :param job_args:
    :return:
    """
    from alibaba_ai_task.train.trainer import APOTrainer
    from alibaba_ai_task.train.data_module import APODATAModule

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint
    persistent_workers = True,

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import CSVLogger

    from pytorch_lightning.plugins import DDPPlugin

    import glob
    import os
    import os.path as osp
    import sys
    from tqdm import tqdm

    import torch
    from alibaba_ai_task.tools.omni_tools import makepath
    from loguru import logger

    cfg = APOTrainer.prepare_train_cfg(**job_args)

    if cfg.trainer.deterministic:
        pl.seed_everything(cfg.trainer.rnd_seed, workers=True)

    log_format = f"{{module}}:{{function}}:{{line}} -- {cfg.apo.expr_id} -- {cfg.apo.data_id} -- {{message}}"

    logger.remove()

    logger.add(cfg.dirs.log_fname, format=log_format, enqueue=True)
    logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

    dm = APODATAModule(cfg)
    dm.prepare_data()
    dm.setup(stage='fit')

    model = APOTrainer(cfg)

    makepath(cfg.dirs.log_dir, 'tensorboard')
    makepath(cfg.dirs.log_dir, 'csv')
    tboard_logger = TensorBoardLogger(cfg.dirs.log_dir, name='tensorboard')
    csv_logger = CSVLogger(cfg.dirs.log_dir, name='csv', version=None, prefix='')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    snapshots_dir = osp.join(model.work_dir, 'snapshots')
    checkpoint_callback = ModelCheckpoint(
        dirpath=makepath(snapshots_dir, isfile=True),
        filename="%s_{epoch:02d}_{val_loss:.2f}" % model.cfg.apo.expr_id,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    early_stop_callback = EarlyStopping(**model.cfg.train_parms.early_stopping)

    resume_checkpoint_fname = cfg.trainer.resume_checkpoint_fname
    if cfg.trainer.resume_training_if_possible and not cfg.trainer.fast_dev_run:
        if not resume_checkpoint_fname:
            available_ckpts = sorted(glob.glob(osp.join(snapshots_dir, '*.ckpt')), key=os.path.getmtime)
            if len(available_ckpts) > 0:
                resume_checkpoint_fname = available_ckpts[-1]
    if resume_checkpoint_fname:
        logger.info(f'Resuming the training from {resume_checkpoint_fname}')

    if cfg.trainer.finetune_checkpoint_fname:  # only reuse weights and not the learning rates
        state_dict = torch.load(cfg.trainer.finetune_checkpoint_fname)['state_dict']
        model.load_state_dict(state_dict, strict=True)
        # Todo fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
        logger.info(f'Loaded finetuning weights from {cfg.trainer.finetune_checkpoint_fname}')

    trainer = pl.Trainer(gpus=1 if cfg.trainer.fast_dev_run else cfg.trainer.num_gpus,
                         weights_summary=cfg.trainer.weights_summary,
                         # distributed_backend=None if cfg.trainer.fast_dev_run else cfg.trainer.distributed_backend,
                         strategy=None if cfg.trainer.strategy=='ddp' else cfg.trainer.strategy,
                         profiler=cfg.trainer.profiler,
                         plugins=None if cfg.trainer.fast_dev_run and not cfg.trainer.strategy=='ddp'  else [DDPPlugin(find_unused_parameters=False)],
                         fast_dev_run=cfg.trainer.fast_dev_run,
                         limit_train_batches=cfg.trainer.limit_train_batches,
                         limit_val_batches=cfg.trainer.limit_val_batches,
                         num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,  # run full validation run
                         callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
                         max_epochs=cfg.trainer.max_epochs,
                         logger=[tboard_logger, csv_logger],
                         resume_from_checkpoint=resume_checkpoint_fname,
                         deterministic=cfg.trainer.deterministic,
                         overfit_batches=cfg.trainer.overfit_batches,

                         )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
