#include <fstream>
#include <string>
#include <memory>
#include <algorithm>
#include "Filter.h"

using std::string;
int main(int argc, char **argv)
{
	if (argc != 7)
	{
		std::cerr << "This program is used to select candidate from distance sequence.\n"
			<< "usage: test dir_name video_list all_rates filter_type file_type output_dir\n"
			<< "dir_name:包含不同采样率上的距离文件的目录路径\n"
			<< "video_list:包含所有要处理的视频名称的文件\n"
			<< "all_rates:用逗号隔开的采样率序列"
			<< "filter_type: 0:DSM 1:DSM-edit1 2:DSM-edit2 3:Myfilter\n"
			<<"file_type: 距离文件的类型 0-文本文件 1-二进制文件\n"
			<< "output_dir: 输出的候选帧结果文件的保存路径" <<std::endl;
		exit(1);
	}
	int pos_number = 0;
	string dir_name(argv[++pos_number]);
	if (dir_name.back() != '\\' && dir_name.back() != '/')
		dir_name.push_back('/');
	string video_list_file(argv[++pos_number]);
	
	string rates_str(argv[++pos_number]);
	vector<int> all_rates;
	
	string::size_type sub_begin = 0;
	string::size_type pos = rates_str.find(',', sub_begin);
	int rate = 0;
	while (pos != string::npos)
	{
		rate = std::stoi(rates_str.substr(sub_begin, pos));
		sub_begin = pos + 1;
		pos = rates_str.find(',', sub_begin);
		all_rates.push_back(rate);
	}
	rate = std::stoi(rates_str.substr(sub_begin, pos));
	all_rates.push_back(rate);

	int filter_type = std::atoi(argv[++pos_number]);
	int file_type = std::atoi(argv[++pos_number]);
	string output_dir(argv[++pos_number]);
	if (output_dir.back() != '\\' && output_dir.back() != '/')
		output_dir.push_back('/');
	std::sort(all_rates.begin(), all_rates.end());	//确保采样率是递增的
	//设置过滤算法的超参数
	double static_threshold = 0.2, sigma = 0.05;
	double a = 0.7;
	if (filter_type == 0 || filter_type == 1 || filter_type == 2)
	{
		std::cout << "Enter the static threshold: ";
		std::cin >> static_threshold;
		std::cout << "Enter the hyperParameter sigma: ";
		std::cin >> sigma;
	}else
		if (filter_type == 3)
		{
			std::cout << "Enter the hyperParameter a: ";
			std::cin >> a;
		}
	std::ifstream video_list(video_list_file);
	if (video_list.is_open())
	{
		string video_name;
		while (std::getline(video_list, video_name))
		{
			if (filter_type < 3)
			{
				std::shared_ptr<DSM_Filter> filter = getDSM_Filter(filter_type,dir_name + video_name, all_rates, file_type);
				filter->set_static_threshold(static_threshold);
				filter->set_sigma(sigma);
				std::cout << "Using Filter " << filter->type() << std::endl;
				std::cout << "static_threshold: " << filter->get_static_threshold() << " sigma: " << filter->get_sigma() << std::endl;
				std::cout << "Window size: " << filter->get_window_size() << std::endl;
				filter->filter();
				filter->save_result(output_dir + video_name + "_candidates");
			}
			else if (filter_type == 3)
			{
				MyFilter filter(dir_name + video_name, all_rates, file_type);
				filter.set_a(a);
				std::cout << "Using Filter " << filter.type() << std::endl;
				std::cout << "hyperParameter a: " << filter.get_a() << std::endl;
				std::cout << "Window size: " << filter.get_window_size() << std::endl;
				filter.filter();
				filter.save_result(output_dir + video_name + "_candidates");
			}
		}
	}
	else
	{
		std::cerr << "cannot open the video list file " << video_list_file << std::endl;
		return 1;
	}
	
	return 0;
}