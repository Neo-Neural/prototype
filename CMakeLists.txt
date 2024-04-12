cmake_minimum_required(VERSION 3.16)
project(prototype)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ext/fmt)
find_package(fmt)

add_subdirectory(ext/spdlog)
find_package(spdlog)

add_executable(prototype main.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/fmt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/spdlog/include)


target_link_libraries(
  prototype
  PRIVATE
    fmt::fmt
)