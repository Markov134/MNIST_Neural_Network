cmake_minimum_required(VERSION 3.31)
project(Neuron_Network_Image_Detection)

set(CMAKE_CXX_STANDARD 23)

add_executable(Neuron_Network_Image_Detection main.cpp
        src/DataLoader.cpp
        src/Data.cpp
        src/Net.cpp)
