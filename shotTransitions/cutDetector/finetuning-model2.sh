log_dir=results_final

#对kinetics上预训练的模型进行finetuing
nohup python -u main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train+val_samples_1_6 \
--no_val \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 2 \
--n_finetune_classes 2 \
--pretrain_path pretrain_models/model_epoch2.pth \
--ft_begin_index 2 \
--sample_size 112 \
--sample_duration 16 \
--batch_size 64 \
--n_epochs 100 \
--auto_resume \
--train_subdir train \
--model resnet \
--model_depth 50 \
--n_threads 20 \
--learning_rate 0.001 \
--weight_decay 1e-5 \
--lr_step 5 \
--lr_patience 5 \
--checkpoint 1  |tee  data/$log_dir/screen_1.log &
