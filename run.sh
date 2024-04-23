python pcnn_train.py \
--batch_size 8 \
--sample_batch_size 16 \
--sampling_interval 50 \
--save_interval 50 \
--dataset cpen455 \
--nr_resnet 2 \
--nr_filters 80 \
--nr_logistic_mix 10 \
--lr_decay 0.999995 \
--max_epochs 500 \
--en_wandb True \
