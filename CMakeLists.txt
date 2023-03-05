cmake_minimum_required(VERSION 3.0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin/${CMAKE_BUILD_TYPE})

set(T_LIB_NAME "ft")
set(T_PROJECT_NAME "lab_4")

project(${T_PROJECT_NAME} CXX)

find_package(OpenCV REQUIRED)

set(HEADERS ft/ft.hpp)
set(SOURCES ft/ft.cpp)

add_library(${T_LIB_NAME} STATIC ${HEADERS} ${SOURCES})

add_executable(${T_PROJECT_NAME} main.cpp)

target_link_libraries(${T_LIB_NAME} PUBLIC ${OpenCV_LIBS})

target_link_libraries(${T_PROJECT_NAME} PRIVATE ${T_LIB_NAME})

