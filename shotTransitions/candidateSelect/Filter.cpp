#include "Filter.h"
#include <limits>
#include <algorithm>
#include <cmath>

Filter::Filter(const string & distance_file_prefix, const vector<int>& all_rates,int file_type)
	:window_size(16), min_space(5),lamda(3),gamma(0.8)
{
	for (size_t i = 0; i < all_rates.size(); ++i)
	{
		string file_name(distance_file_prefix);
		file_name.push_back('_');
		file_name += std::to_string(all_rates[i]);
		
		vector<pair<int, double>> temp_buffer; 
		if (file_type == 0)	//文本文件类型
			load_from_txt(file_name, temp_buffer);
		else
			load_from_bin(file_name, temp_buffer);
		distances_at_all_rates.push_back(temp_buffer);
		
	}
}

Filter::Filter(const vector<vector<pair<int, double>>>& all_distances_at_rates)
	: window_size(16),min_space(5),lamda(3),gamma(0.8),distances_at_all_rates(all_distances_at_rates)
{
}

void Filter::filter()
{
	vector<vector<int>> candidates_at_all_rates;
	for (size_t i = 0; i < distances_at_all_rates.size(); ++i)
	{
		vector<int> temp = filter_core(distances_at_all_rates[i]);
		candidates_at_all_rates.push_back(temp);
	}
	//合并不同采样率的结果
	merge_candidates(candidates_at_all_rates);
}

//输出结果，path为结果文件的路径
void Filter::save_result(const string & path)
{
	std::ofstream output(path);
	if (!output.is_open())
	{
		std::cerr << "fail to save candidates, "
			<< "cannot open " << path << std::endl;
		exit(1);
	}
	for (auto candidate : all_candidates)
	{
		output << candidate << std::endl;
	}
	output.close();
}
//合并不同采样率得到的candidate
void Filter::merge_candidates(vector<vector<int>>& candidates_at_all_sampleRates)
{
	if (candidates_at_all_sampleRates.empty())
		return;

	std::list<int> temp(candidates_at_all_sampleRates[0].begin(), candidates_at_all_sampleRates[0].end());
	for (size_t i = 1; i < candidates_at_all_sampleRates.size(); ++i)
	{

		auto head = temp.begin();	//指向低采样率时得到的候选段的头部
		auto back = temp.end();//指向低采样率时得到的候选段的
		auto it = head;
		auto prev_it = back;
		//遍历采样率i的所有candidates
		for (size_t j = 0; j < candidates_at_all_sampleRates[i].size(); ++j)
		{
			//在temp中找到合适的插入位置
			while (it != temp.end() && *it < candidates_at_all_sampleRates[i][j])
			{
				prev_it = it;
				++it;
			}
			//如果不同采样率的candidates相隔太近，则只保留低采样率的candidates
			if (it == head)
			{
				if (it == back || *it >= candidates_at_all_sampleRates[i][j] + min_space)
				{
					temp.insert(it, candidates_at_all_sampleRates[i][j]);
				}

			}
			else if (it == back)
			{
				if (candidates_at_all_sampleRates[i][j] >= *prev_it + min_space)
					temp.insert(it, candidates_at_all_sampleRates[i][j]);
			}
			else
			{
				if (*it >= candidates_at_all_sampleRates[i][j] + min_space
					&& candidates_at_all_sampleRates[i][j] >= *prev_it + min_space)
					temp.insert(it, candidates_at_all_sampleRates[i][j]);
			}


		}
	}
	all_candidates.resize(temp.size());
	std::copy(temp.begin(), temp.end(), all_candidates.begin());
}
//从文本文件中加载距离序列
void Filter::load_from_txt(const string & file_name, vector<pair<int, double>>& distance_sequence)
{
	std::ifstream input(file_name);
	if (!input.is_open())
	{
		std::cerr << "cannot open distance file " << file_name
			<< " please check!!!" << std::endl;
		exit(1);
	}
	string line;
	while (std::getline(input, line))
	{
		std::stringstream line_input(line);
		int frame_no = -1;
		double distance = 0;
		line_input >> frame_no >> distance;
		distance_sequence.push_back(std::make_pair(frame_no, distance));
	}
}
//从二进制文件中加载距离序列
void Filter::load_from_bin(const string & file_name, vector<pair<int, double>>& distance_sequence)
{
	std::ifstream input(file_name,std::ios::binary);
	if (!input.is_open())
	{
		std::cerr << "cannot open distance file " << file_name
			<< " please check!!!" << std::endl;
		exit(1);
	}
	distance_output::video_sequence msg;
	if (!msg.ParseFromIstream(&input))
	{
		std::cerr << "Failed to parse input distances file " << file_name << std::endl;
		exit(1);
	}
	int num = msg.per_frame_size();
	for (int i = 0; i < num; ++i)
	{
		int frame_no = msg.per_frame(i).frame_no();
		double value = msg.per_frame(i).value();
		distance_sequence.push_back(std::make_pair(frame_no, value));
	}
}
DSM_Filter::DSM_Filter(const string & distance_file_prefix, const vector<int>& all_rates,int file_type)
	:Filter(distance_file_prefix,all_rates,file_type),sigma(0.05), static_threshold(0.5)
{
}
DSM_Filter::DSM_Filter(const vector<vector<pair<int, double>>>& all_distances_at_rates)
	:Filter(all_distances_at_rates), sigma(0.05), static_threshold(0.5)
{
}
vector<int> DSM_Filter::filter_core(const vector<pair<int, double>>& distances)
{
	int window_begin = 0;
	int window_end = 0;	//指向滑动窗口外的下一个数据

	double local_mean = 0.0;		//滑动窗口中的均值
	double local_sum = 0.0;
	vector<int> candidates;
	if (distances.empty())
		return candidates;
	double global_mean = 0.0;
	for (size_t i = 0; i < distances.size(); ++i)
		global_mean += distances[i].second;
	global_mean /= distances.size();
	for (int i = 0; i < distances.size(); ++i)
	{
		//计算以该帧图像为中心的滑动窗中的局部均值
		//D(i-window_size+1),D(i-window_size+2),...D(i+window_size-1)
		while (window_end <= i + window_size - 1 && window_end < distances.size())
		{
			local_sum += distances[window_end].second;
			++window_end;
		}
		local_mean = local_sum / (window_end - window_begin);
		double threshold = static_threshold + sigma * local_mean;
		if (distances[i].second > threshold)
			candidates.push_back(distances[i].first);
		/*else
		{
			//如果该帧的距离值比neighbor的大的多
			if (((i > 0 && distances[i].second > lamda * distances[i - 1].second)
				|| (i + 1< distances.size() && distances[i].second > lamda * distances[i + 1].second))
				&& distances[i].second > gamma*global_mean)
				candidates.push_back(distances[i].first);
		}*/
		while (window_begin >= 0 && window_begin <= i - window_size + 1)
		{
			local_sum -= distances[window_begin].second;
			++window_begin;
		}
	}
	return candidates;
}

//T = static_threshold + sigma * (local_mean + local_deviation *global_mean /(local_mean + epsilon))
vector<int> DSM_Filter_edit1::filter_core(const vector<pair<int, double>>& distances)
{
	int window_begin = 0;
	int window_end = 0;	//指向滑动窗口外的下一个数据

	double local_mean = 0.0;		//滑动窗口中的均值
	double local_sum = 0.0;
	double local_square_sum = 0.0;
	vector<int> candidates;
	if (distances.empty())
		return candidates;
	double global_mean = 0.0;
	for (size_t i = 0; i < distances.size(); ++i)
		global_mean += distances[i].second;
	global_mean /= distances.size();
	for (int i = 0; i < distances.size(); ++i)
	{
		//计算以该帧图像为中心的滑动窗中的局部均值
		//D(i-window_size+1),D(i-window_size+2),...D(i+window_size-1)
		while (window_end <= i + window_size - 1 && window_end < distances.size())
		{
			local_sum += distances[window_end].second;
			local_square_sum += distances[window_end].second * distances[window_end].second;
			++window_end;
		}
		local_mean = local_sum / (window_end - window_begin);
		double local_d = local_square_sum - 2 * local_mean * local_sum + (window_end - window_begin)*local_mean*local_mean;
		local_d = std::sqrt(local_d / (window_end - window_begin - 1));
		
		double threshold = static_threshold + sigma * (local_mean + local_d *global_mean /( local_mean + std::numeric_limits<double>::epsilon()));
		if (distances[i].second > threshold)
			candidates.push_back(distances[i].first);
		/*else
		{
			//如果该帧的距离值比neighbor的大的多
			if (((i > 0 && distances[i].second > lamda * distances[i - 1].second)
				|| (i + 1< distances.size() && distances[i].second > lamda * distances[i + 1].second))
				&& distances[i].second > gamma*global_mean)
				candidates.push_back(distances[i].first);
		}*/
		while (window_begin >= 0 && window_begin <= i - window_size + 1)
		{
			local_sum -= distances[window_begin].second;
			++window_begin;
		}
	}
	return candidates;
}

//T = static_threshold + sigma * (local_mean + local_deviation *( 1 + ln(global_mean /(local_mean + epsilon))))
vector<int> DSM_Filter_edit2::filter_core(const vector<pair<int, double>>& distances)
{
	int window_begin = 0;
	int window_end = 0;	//指向滑动窗口外的下一个数据

	double local_mean = 0.0;		//滑动窗口中的均值
	double local_sum = 0.0;
	double local_square_sum = 0.0;
	vector<int> candidates;
	if (distances.empty())
		return candidates;
	double global_mean = 0.0;
	for (size_t i = 0; i < distances.size(); ++i)
		global_mean += distances[i].second;
	global_mean /= distances.size();
	for (int i = 0; i < distances.size(); ++i)
	{
		//计算以该帧图像为中心的滑动窗中的局部均值
		//D(i-window_size+1),D(i-window_size+2),...D(i+window_size-1)
		while (window_end <= i + window_size - 1 && window_end < distances.size())
		{
			local_sum += distances[window_end].second;
			local_square_sum += distances[window_end].second * distances[window_end].second;
			++window_end;
		}
		local_mean = local_sum / (window_end - window_begin);
		double local_d = local_square_sum - 2 * local_mean * local_sum + (window_end - window_begin)*local_mean*local_mean;
		local_d = std::sqrt(local_d / (window_end - window_begin - 1));

		double threshold = static_threshold + sigma * (local_mean + local_d *( 1+ std::log(global_mean / (local_mean + std::numeric_limits<double>::epsilon()))));
		if (distances[i].second > threshold)
			candidates.push_back(distances[i].first);
		/*else
		{
			//如果该帧的距离值比neighbor的大的多
			if (((i > 0 && distances[i].second > lamda * distances[i - 1].second)
				|| (i + 1< distances.size() && distances[i].second > lamda * distances[i + 1].second))
				&& distances[i].second > gamma*global_mean)
				candidates.push_back(distances[i].first);
		}*/
		while (window_begin >= 0 && window_begin <= i - window_size + 1)
		{
			local_sum -= distances[window_begin].second;
			++window_begin;
		}
	}
	return candidates;
}

vector<int> MyFilter::filter_core(const vector<pair<int, double>>& distances)
{
	int window_begin = 0;
	int window_end = 0;	//指向滑动窗口外的下一个数据

	double local_mean = 0.0;		//滑动窗口中的均值
	double local_sum = 0.0;
	double local_square_sum = 0.0;
	vector<int> candidates;
	if (distances.empty())
		return candidates;
	double global_mean = 0.0;
	for (size_t i = 0; i < distances.size(); ++i)
		global_mean += distances[i].second;
	global_mean /= distances.size();
	for (int i = 0; i < distances.size(); ++i)
	{
		//计算以该帧图像为中心的滑动窗中的局部均值
		//D(i-window_size+1),D(i-window_size+2),...D(i+window_size-1)
		while (window_end <= i + window_size - 1 && window_end < distances.size())
		{
			local_sum += distances[window_end].second;
			local_square_sum += distances[window_end].second * distances[window_end].second;
			++window_end;
		}
		local_mean = local_sum / (window_end - window_begin);
		double local_d = local_square_sum - 2 * local_mean * local_sum + (window_end - window_begin)*local_mean*local_mean;
		local_d = std::sqrt(local_d / (window_end - window_begin - 1));

		double threshold = local_mean + a * local_d *(1 + std::log(global_mean / (local_mean + std::numeric_limits<double>::epsilon())));
        //如果该帧的距离值比neighbor的大的多
		if (((i > 0 && distances[i].second > lamda * distances[i - 1].second)
			|| (i + 1< distances.size() && distances[i].second > lamda * distances[i + 1].second))
			&& distances[i].second > gamma*global_mean)
			candidates.push_back(distances[i].first);
        /*else{
            if (distances[i].second > threshold)
                candidates.push_back(distances[i].first);
        }*/
		while (window_begin >= 0 && window_begin <= i - window_size + 1)
		{
			local_sum -= distances[window_begin].second;
			++window_begin;
		}
	}
	return candidates;
}

std::shared_ptr<DSM_Filter> getDSM_Filter(int filter_type, const string & distance_file_prefix, const vector<int>& all_rates, int file_type)
{
	if (filter_type == 0)
		return std::shared_ptr<DSM_Filter>(new DSM_Filter(distance_file_prefix, all_rates, file_type));
	else if (filter_type == 1)
		return std::shared_ptr<DSM_Filter>(new DSM_Filter_edit1(distance_file_prefix, all_rates, file_type));
	else if (filter_type == 2)
		return std::shared_ptr<DSM_Filter>(new DSM_Filter_edit2(distance_file_prefix, all_rates, file_type));
    else if (filter_type == 4)
		return std::shared_ptr<DSM_Filter>(new DSM_Filter_edit3(distance_file_prefix, all_rates, file_type));
	else
		return nullptr;
}
//T = static_threshold + sigma * local_mean*( 1+ ln(global_mean /(local_mean + epsilon)))
vector<int> DSM_Filter_edit1::filter_core(const vector<pair<int, double>>& distances)
{
	int window_begin = 0;
	int window_end = 0;	//指向滑动窗口外的下一个数据

	double local_mean = 0.0;		//滑动窗口中的均值
	double local_sum = 0.0;
	double local_square_sum = 0.0;
	vector<int> candidates;
	if (distances.empty())
		return candidates;
	double global_mean = 0.0;
	for (size_t i = 0; i < distances.size(); ++i)
		global_mean += distances[i].second;
	global_mean /= distances.size();
	for (int i = 0; i < distances.size(); ++i)
	{
		//计算以该帧图像为中心的滑动窗中的局部均值
		//D(i-window_size+1),D(i-window_size+2),...D(i+window_size-1)
		while (window_end <= i + window_size - 1 && window_end < distances.size())
		{
			local_sum += distances[window_end].second;
			local_square_sum += distances[window_end].second * distances[window_end].second;
			++window_end;
		}
		local_mean = local_sum / (window_end - window_begin);
		//double local_d = local_square_sum - 2 * local_mean * local_sum + (window_end - window_begin)*local_mean*local_mean;
		//local_d = std::sqrt(local_d / (window_end - window_begin - 1));
		
		double threshold = static_threshold + sigma * local_mean*( 1+ std::log(global_mean /(local_mean + epsilon)));
		if (distances[i].second > threshold)
			candidates.push_back(distances[i].first);
		/*else
		{
			//如果该帧的距离值比neighbor的大的多
			if (((i > 0 && distances[i].second > lamda * distances[i - 1].second)
				|| (i + 1< distances.size() && distances[i].second > lamda * distances[i + 1].second))
				&& distances[i].second > gamma*global_mean)
				candidates.push_back(distances[i].first);
		}*/
		while (window_begin >= 0 && window_begin <= i - window_size + 1)
		{
			local_sum -= distances[window_begin].second;
			++window_begin;
		}
	}
	return candidates;
}
