#!/bin/sh
# Run script for multiple runs

# source /home/$USER/.bashrc
# precip
python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1" --data.fold 0 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_notime" \
--trainer.logger.version "fold0" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1" --data.fold 1 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_notime" \
--trainer.logger.version "fold1" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1" --data.fold 2 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_notime" \
--trainer.logger.version "fold2" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1" --data.fold 3 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_notime" \
--trainer.logger.version "fold3" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1" --data.fold 4 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_notime" \
--trainer.logger.version "fold4" --trainer.logger.default_hp_metric false



# nocorr
python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1_nocorr+time" --data.fold 0 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_nocorr+time" \
--trainer.logger.version "fold0" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1_nocorr+time" --data.fold 1 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_nocorr+time" \
--trainer.logger.version "fold1" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1_nocorr+time" --data.fold 2 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_nocorr+time" \
--trainer.logger.version "fold2" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1_nocorr+time" --data.fold 3 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_nocorr+time" \
--trainer.logger.version "fold3" --trainer.logger.default_hp_metric false

python /home/gregor/precipitation/eva_precipitation/precipitation/precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.devices [1] --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "gpu" \
--data.feature_set "v1_nocorr+time" --data.fold 4 --data.data_dir "/home/gregor/datasets/precipitation" \
--trainer.logger.class_path "pytorch_lightning.loggers.TensorBoardLogger" --trainer.logger.save_dir "/home/gregor/precipitation/eva_precipitation/logs" --trainer.logger.name "UNet-64-0.2_nocorr+time" \
--trainer.logger.version "fold4" --trainer.logger.default_hp_metric false
