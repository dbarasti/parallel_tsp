cmake_minimum_required(VERSION 3.15)
project(tsp_ga)

set(CMAKE_CXX_STANDARD 20)

add_executable(tsp_ga main.cpp Point.h)

add_subdirectory(Google_tests)