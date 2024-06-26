cmake_minimum_required(VERSION 3.16)
project(prototype)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ext/fmt)
add_subdirectory(ext/spdlog)
add_subdirectory(ext/json)
add_subdirectory(ext/armadillo)

add_executable(prototype main.cpp)

include_directories(ext/fmt/include)
include_directories(ext/spdlog/include)
include_directories(ext/json/include)
include_directories(ext/armadillo/include)
include_directories(include)

file(GLOB_RECURSE SOURCES src/*.cpp)
target_sources(prototype PRIVATE ${SOURCES})

target_link_libraries(
  prototype
  PRIVATE
    fmt::fmt
)