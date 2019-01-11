#include <cmath>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <cstring>
#include <vector>
using std::string;
class MeanStdMeter{
private:
    long long count;
    double m;
    double v;
    std::mutex mutex_lock;
public:
    MeanStdMeter():count(0),m(0),v(0){}
    void reset();
    void update(int val);
    double get_mean(){return m / count;}
    //double get_std();
};

void MeanStdMeter::reset()
{
    std::lock_guard<std::mutex> lock(mutex_lock);
    m = 0;
    //v = 0;
    count = 0;
}
void MeanStdMeter::update(int val)
{
    std::lock_guard<std::mutex> lock(mutex_lock);
    ++count;
    /*
    if(count == 1)
        v = 0.0;

    else
    {
        v += (m - (count - 1)* val) *(m - (count - 1) * val) /(count * (count - 1));
    }*/
    m += val;
}

/*
double MeanStdMeter::get_std()
{
    if(count == 1)
        return 1;

    else
    {
        return std::sqrt(v /(count -1));
    }

}*/

std::mutex g_fd_mutex;  //protects global fd
MeanStdMeter r_avg,g_avg,b_avg;

void thread_funct(std::shared_ptr<std::ifstream> video_list_fd)
{
    std::cout << "thread start." << std::endl;
    std::unique_lock<std::mutex> file_lock(g_fd_mutex,std::defer_lock);
    string video_path;
    while(true)
    {
        file_lock.lock();
        //if(std::getline(*video_list_fd,video_path))
        if((*video_list_fd) >> video_path)
        {
            file_lock.unlock();

            cv::VideoCapture cap(video_path);
            if(!cap.isOpened())
            {
                std::cerr << "Cannot open " << video_path;
                continue;
            }
            cv::Mat img;
            cap >> img;
            while(!img.empty())
            {
                int rows = img.rows;
                int cols = img.cols;
                for(int row = 0; row < rows; ++row)
                {
                    const uchar *ptr = img.ptr<uchar>(row);
                    for(int col = 0; col < cols;++col)
                    {
                        b_avg.update(*ptr++);
                        g_avg.update(*ptr++);
                        r_avg.update(*ptr++);
                    }
                }
                cap >> img;
            }
            std::cout << "Finished Processing " << video_path << std::endl;

        }else
        {
            file_lock.unlock();
            break;
        }

    }
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "Usage: cal_mean videos_list_path thread_num" << std::endl;
        exit(1);
    }

    string video_list_path(argv[1]);
    int thread_num = std::atoi(argv[2]);
    if(thread_num < 1)
    {
        thread_num = 1;
    }
    std::shared_ptr<std::ifstream> video_list_fd = std::make_shared<std::ifstream>(video_list_path);

    std::vector<std::shared_ptr<std::thread>> exec_units;
    for(int i = 0; i < thread_num;++i)
    {
        std::shared_ptr<std::thread> t = std::make_shared<std::thread>(thread_funct,video_list_fd);
        exec_units.push_back(t);
    }
    for(int i = 0; i < thread_num;++i)
    {
        exec_units[i]->join();
    }
    std::cout << "Finished" << std::endl;
    std::cout << "The mean values of pixels: R: " << r_avg.get_mean() << " G: " << g_avg.get_mean() << " B: " << b_avg.get_mean() <<std::endl;
    //std::cout << "The standard deviation of pixels: R: " << r_avg.get_std() << " G: " << g_avg.get_std() << " B: " << b_avg.get_std() <<std::endl;
    return 0;
}