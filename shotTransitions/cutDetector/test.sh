log_dir=results

#根据训练好的模型文件来在测试集上进行测试
nohup python -u main.py \
--no_train \
--no_val \
--test \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--test_list_path test_samples \
--result_path $log_dir \
--n_classes 2 \
--sample_size 128 \
--sample_duration 6 \
--batch_size 128 \
--resume_path results/model_epoch15.pth \
--test_subdir test \
--model xcresnet \
--model_depth 50 \
--n_threads 12 |tee  data/$log_dir/screen.log 
