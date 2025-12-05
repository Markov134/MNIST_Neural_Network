cmake_minimum_required(VERSION 3.31)
project(Neuron_Network_Image_Detection)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(NerualNetTest Tests/neuralNetTest.cpp
        src/DataLoader.cpp
        src/Data.cpp
        src/Net.cpp)
add_executable(Neuron_Network_Image_Detection main.cpp
        src/DataLoader.cpp
        src/Data.cpp
        src/Net.cpp)

target_link_libraries(NerualNetTest PRIVATE Catch2::Catch2WithMain)