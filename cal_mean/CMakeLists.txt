cmake_minimum_required(VERSION 3.13)
project(cal_mean)

set(CMAKE_CXX_STANDARD 11)

add_executable(cal_mean main.cpp)
target_link_libraries(cal_mean opencv_videoio
        /lib/x86_64-linux-gnu/libpthread.so.0
        /usr/local/lib/libopencv_core.so)