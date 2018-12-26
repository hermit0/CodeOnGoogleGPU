#ifndef _FILTER_H_
#define _FILTER_H_

#include <vector>
#include <string>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>

#include "distance_output.pb.h"

using std::list;
using std::pair;
using std::vector;
using std::string;
//对输入的不同采样的距离序列执行DSM中的Filter算法
class Filter
{
public:
	//提供距离文件的公共前缀和采样率作为算法的输入，距离文件的全名为 公共前缀_采样率 距离文件的类型：0-文本 非零值-二进制
	Filter(const string &distance_file_prefix, const vector<int> &all_rates, int file_type = 0);	
	
	Filter(const vector<vector<pair<int, double>>> &all_distances_at_rates);//提供不同采样率的距离序列集合作为算法输入
	void set_window_size(int a) { window_size = a; }
	int get_window_size() { return window_size; }
	void set_min_space(int a) { min_space = a; }
	int get_min_space() { return min_space; }
    void set_lamda(double val){lamda=val;}
    double get_lamda(){return lamda;}
    void set_gamma(double val){gamma = val;}
    double get_gamma(){return gamma;}
	void filter();	//执行过滤算法
	void save_result(const string &path);//输出结果
	virtual string type() = 0;
	virtual ~Filter() {}
protected:
	virtual vector<int> filter_core(const vector<pair<int, double>> &distances)=0;	//过滤单个序列
private:
	void merge_candidates(vector<vector<int>> &candidates_at_all_sampleRates);//合并不同采样率得到的candidate
	void load_from_txt(const string &file_name, vector<pair<int, double>> &distance_sequence);
	void load_from_bin(const string &file_name, vector<pair<int, double>> &distance_sequence);
protected:
	int window_size;	//窗口的一半大小
	int min_space;	//不同采样率候选帧之间的最小间隔
	double lamda,gamma; //邻居比较时的超参数
	vector<vector<pair<int, double>>> distances_at_all_rates;
	vector<int> all_candidates;
};
class DSM_Filter : public Filter {
public:
	DSM_Filter(const string &distance_file_prefix, const vector<int> &all_rates,int file_type = 0);	//提供距离文件的公共前缀和采样率作为算法的输入，距离文件的全名为 公共前缀_采样率
	DSM_Filter(const vector<vector<pair<int, double>>> &all_distances_at_rates);//提供不同采样率的距离序列集合作为算法输入
	void set_sigma(double value) { sigma = value; }
	double get_sigma() { return sigma; }
	void set_static_threshold(double t) { static_threshold = t; }
	double get_static_threshold() { return static_threshold; }
	string type() { return "DSM"; }
protected:
	virtual vector<int> filter_core(const vector<pair<int, double>> &distances);	//过滤单个序列
	double sigma;
	double static_threshold;
};
class DSM_Filter_edit1 :public DSM_Filter {
public:
	DSM_Filter_edit1(const string &distance_file_prefix, const vector<int> &all_rates, int file_type = 0)
		:DSM_Filter(distance_file_prefix, all_rates,file_type) {}
	DSM_Filter_edit1(const vector<vector<pair<int, double>>> &all_distances_at_rates)
		:DSM_Filter(all_distances_at_rates) {}
	string type() { return "DSM_Filter_edit1"; }
protected:
	virtual vector<int> filter_core(const vector<pair<int, double>> &distances);	//过滤单个序列
};

class DSM_Filter_edit2 :public DSM_Filter {
public:
	DSM_Filter_edit2(const string &distance_file_prefix, const vector<int> &all_rates, int file_type = 0)
		:DSM_Filter(distance_file_prefix, all_rates, file_type) {}
	DSM_Filter_edit2(const vector<vector<pair<int, double>>> &all_distances_at_rates)
		:DSM_Filter(all_distances_at_rates) {}
	string type() { return "DSM_Filter_edit2"; }
protected:
	virtual vector<int> filter_core(const vector<pair<int, double>> &distances);	//过滤单个序列
};

class MyFilter : public Filter {
public:
	MyFilter(const string &distance_file_prefix, const vector<int> &all_rates, int file_type = 0)
		:Filter(distance_file_prefix, all_rates, file_type), a(0.7) {}

	MyFilter(const vector<vector<pair<int, double>>> &all_distances_at_rates)
		:Filter(all_distances_at_rates), a(0.7) {}
	void set_a(double val) { a = val; }
	double get_a() { return a; }
	string type() { return "MyFilter"; }
protected:
	virtual vector<int> filter_core(const vector<pair<int, double>> &distances);	//过滤单个序列
private:
	double a;	//阈值表达式中的常数比例
};

std::shared_ptr<DSM_Filter> getDSM_Filter(int filter_type, const string &distance_file_prefix, const vector<int> &all_rates, int file_type = 0);
#endif