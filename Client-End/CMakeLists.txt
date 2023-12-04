cmake_minimum_required(VERSION 3.20)
project(ClassificationSystem)

set(CMAKE_CXX_STANDARD 14)

add_executable(ClassificationSystem main.cpp Client.cpp SPT.cpp FirstPath.cpp FirstPath.h DKSP.cpp main1.cpp main1.h FirstPath_addMatrix.cpp FirstPath_addMatrix.h DynamicSimilarity.cpp ClassificationSystem.h)
