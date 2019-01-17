log_dir=results

#对kinetics上预训练的模型进行finetuing
nohup python -u main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train_samples \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 400 \
--n_finetune_classes 2 \
--pretrain_path pretrain_models/resnet-50-kinetics.pth \
--ft_begin_index 0 \
--sample_size 128 \
--sample_duration 6 \
--batch_size 128 \
--n_epochs 100 \
--auto_resume \
--train_subdir train \
--model resnet \
--model_depth 50 \
--n_threads 12 \
--learning_rate 0.01 \
--checkpoint 1 2>error.log |tee  data/$log_dir/screen.log
