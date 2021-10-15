cmake_minimum_required(VERSION 3.15)
project(tsp_ga)

# find packages
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -O0 -fsanitize=thread -fstack-check -fstack-protector-strong -fno-omit-frame-pointer -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize") # vectorization info -fopt-info-vec

set(CMAKE_CXX_STANDARD 20)

add_executable(tsp_ga main.cpp)

add_library(GeneticAlgorithms
        STATIC
        src/lib/Point.h
        src/lib/utils.h
        src/parallelTSP.h
        src/sequentialTSP.h)

add_subdirectory(Google_tests)

set_property(TARGET tsp_ga PROPERTY CXX_STANDARD 17)
set_target_properties(GeneticAlgorithms PROPERTIES
        CXX_STANDARD 17
        LINKER_LANGUAGE CXX)

target_link_libraries(tsp_ga
        GeneticAlgorithms
        Threads::Threads
        )