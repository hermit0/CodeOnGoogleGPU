log_dir = results

#对kinetics上预训练的模型进行finetuing
python main.py \
--root_dir ~/data \
--train_list_path train_samples \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 400 \
--n_finetune_classes 2 \
--pretrain_path pretrain_models/resnet-50-kinetics.pth \
-- ft_begin_index 4 \
--sample_size 224 \
--sample_duration 6 \
--batch_size 128 \
--n_epochs 200 \
--auto_resume \
--train_subdir train \
--model resnet \
--model_depth 50 \
--n_threads 4 \
--learning_rate 0.01 \
--shuffle \
--checkpoint 500 |tee  $log_dir/screen.log