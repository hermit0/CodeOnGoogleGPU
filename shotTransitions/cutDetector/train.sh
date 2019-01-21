log_dir=results

#从头开始训练模型或者从最近的检查点恢复训练，提供验证集进行验证
nohup python -u  main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train_samples \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 2 \
--sample_size 128 \
--sample_duration 16 \
--batch_size 32 \
--n_epochs 100 \
--auto_resume \
--train_subdir train \
--model resnet \
--model_depth 50 \
--n_threads 6 \
--learning_rate 0.01 \
--lr_step 20 \
--lr_patience 5 \
--checkpoint 1 2>error.log |tee  data/$log_dir/screen.log & 
