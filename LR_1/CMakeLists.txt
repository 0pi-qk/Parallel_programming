cmake_minimum_required(VERSION 3.25)
project(LR_1)

set(CMAKE_CXX_STANDARD 17)

add_executable(LR_1 main.cpp)

target_link_libraries(LR_1 -fopenmp)