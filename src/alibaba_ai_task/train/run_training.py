# -*- coding: utf-8 -*-
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.11.19
from alibaba_ai_task.train.trainer import train_apo_once
from alibaba_ai_task.tools.omni_tools import get_support_data_dir

support_dir = get_support_data_dir(__file__)

num_gpus = 2
num_cpus = 6

train_apo_once({
        'apo.expr_id': 'V01',
        'apo.data_id': 'V01',

        'dirs.support_base_dir': support_dir,
        'dirs.work_base_dir': '/home/nghorbani/Desktop/alibaba_ai_task',

        'train_parms.batch_size': 16,
        'train_parms.num_workers': num_cpus,

        'model_parms.num_attention_layers': 3,

        'data_parms.history_length': 7,
        'data_parms.future_length': 7,

        'trainer.max_epochs': 500,
        # 'trainer.overfit_batches': 0.1,

        # 'trainer.fast_dev_run': True,
        'trainer.num_gpus': num_gpus,
        'train_parms.optimizer.args.lr': 1e-3,
    },
)

