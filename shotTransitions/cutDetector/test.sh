log_dir = results

#根据训练好的模型文件来在测试集上进行测试
python main.py \
--no_train \
--no_val \
--test \
--root_dir ~/data \
--test_list_path test_samples \
--result_path $log_dir \
--n_classes 2 \
--sample_size 224 \
--sample_duration 6 \
--batch_size 128 \
--resume_path *.pth \
--test_subdir test \
--model xcresnet \
--model_depth 50 \
--n_threads 4 |tee  $log_dir/screen.log