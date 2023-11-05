cmake_minimum_required(VERSION 3.25)
project(LR_3)

set(CMAKE_CXX_STANDARD 17)

add_executable(LR_3 main.cpp)

find_package(MPI REQUIRED)
target_link_libraries(LR_3 PRIVATE MPI::MPI_CXX -fopenmp)