log_dir=results_final

#根据训练好的模型文件来在测试集上进行测试
nohup python -u main.py \
--no_train \
--no_val \
--test \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--test_list_path test_samples_l3.5_g1.2 \
--result_path $log_dir \
--n_classes 2 \
--sample_size 112 \
--sample_duration 16 \
--batch_size 64 \
--resume_path $log_dir/model_epoch4.pth-2 \
--test_subdir test \
--model resnet \
--model_depth 50 \
--n_threads 20 |tee  data/$log_dir/screen_test.log &
