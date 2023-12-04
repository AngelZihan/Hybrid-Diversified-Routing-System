cmake_minimum_required(VERSION 3.17)
project(Project3_Github2)

#set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.73.0 COMPONENTS system filesystem thread REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(.)

add_executable(Project3_Github2
        main.cpp
        graph.cpp
        heap.h
        graph.h
        tools.h
        eKSP.cpp
        eKSPPrune.cpp
        DynamicSimilarity.cpp
        eKSPCompare.cpp)
target_link_libraries(Project3_Github2 ${Boost_LIBRARIES})