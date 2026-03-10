cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)
project(RandGame VERSION 0.1.0 LANGUAGES CXX)
ADD_DEFINITIONS ( -Wall -Wextra -pedantic -O3 )

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(Model STATIC Model.h )
set_target_properties(Model PROPERTIES LINKER_LANGUAGE CXX)

target_compile_features(Model PUBLIC cxx_std_20)
target_link_libraries(Model PUBLIC Eigen3::Eigen)

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_20)
target_link_libraries(main PUBLIC Model)