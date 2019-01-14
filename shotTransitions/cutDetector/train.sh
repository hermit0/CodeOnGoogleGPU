log_dir=results

#从头开始训练模型或者从最近的检查点恢复训练，提供验证集进行验证
python -u  main.py \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--train_list_path train_samples \
--val_list_path val_samples \
--result_path $log_dir \
--n_classes 2 \
--sample_size 128 \
--sample_duration 6 \
--batch_size 128 \
--n_epochs 200 \
--auto_resume \
--train_subdir train \
--model xcresnet \
--model_depth 50 \
--n_threads 8 \
--learning_rate 0.01 \
--checkpoint 1 |tee  data/$log_dir/screen.log
