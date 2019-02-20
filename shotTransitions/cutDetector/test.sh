log_dir=results_3d_1_1

#根据训练好的模型文件来在测试集上进行测试
nohup python -u main.py \
--no_train \
--no_val \
--test \
--root_dir ~/CodeOnGoogleGPU/shotTransitions/cutDetector/data \
--test_list_path test_samples_l_3_g_1 \
--result_path $log_dir \
--n_classes 2 \
--sample_size 112 \
--sample_duration 16 \
--batch_size 64 \
--resume_path $log_dir/model_epoch13.pth \
--test_subdir test \
--model resnet \
--model_depth 50 \
--n_threads 12 |tee  data/$log_dir/screen_test.log &
