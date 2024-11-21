#!/bin/sh
# Run script for multiple runs

# source /home/$USER/.bashrc
# precip
python ../precipitation/models/unet.py fit --model.learning_rate 0.01 --model.initial_filter_size 64 --model.weight_decay 1.e-8 --model.dropout 0.2 \
--trainer.max_epochs 100 --trainer.log_every_n_steps 50 --trainer.num_sanity_val_steps -1 --trainer.accelerator "cpu" \
--data.feature_set "v2+time" --data.fold 0 --data.data_dir "/home/precip_data" \
--trainer.logger.class_path "lightning.pytorch.loggers.TensorBoardLogger" --trainer.logger.save_dir "./logs" --trainer.logger.name "UNet-64-0.2" \
--trainer.logger.version "fold0" --trainer.logger.default_hp_metric false
