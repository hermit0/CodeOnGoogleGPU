log_dir=results

#对kinetics上预训练的模型进行finetuing
nohup python -u main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train_samples \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 400 \
--n_finetune_classes 2 \
--pretrain_path pretrain_models/resnext-101-kinetics.pth \
--ft_begin_index 4 \
--sample_size 112 \
--sample_duration 16 \
--batch_size 256 \
--n_epochs 100 \
--auto_resume \
--train_subdir train \
--model resnext \
--model_depth 101 \
--n_threads 12 \
--learning_rate 0.001 \
--weight_decay 1e-5 \
--checkpoint 1 2>error.log |tee  data/$log_dir/screen.log &
