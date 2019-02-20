log_dir=results_3d_1_1

#从头开始训练模型或者从最近的检查点恢复训练，提供验证集进行验证
nohup python -u  main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train+hard_samples+val \
--no_val \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 2 \
--sample_size 112 \
--sample_duration 16 \
--batch_size 64 \
--n_epochs 100 \
--auto_resume \
--train_subdir train \
--model resnet \
--model_depth 50 \
--n_threads 12 \
--learning_rate 0.01 \
--lr_step 10 \
--lr_patience 5 \
--checkpoint 1 2>error.log |tee  data/$log_dir/screen.log & 
