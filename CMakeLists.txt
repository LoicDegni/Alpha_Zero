cmake_minimum_required(VERSION 3.14)

project(TP3)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)



include(FetchContent)

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip")

FetchContent_Declare(
  libtorch
  URL ${LIBTORCH_URL}
)

FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  message(STATUS "Téléchargement de libtorch...")
  FetchContent_Populate(libtorch)
endif()

# Pointer CMake vers le bon répertoire
list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



file(
    GLOB
    source_files
    main.cpp
    src/*
)

file(
    GLOB
    arene_source_files
    arene.cpp
    src/*
)


set(externLibs
    z
    ncurses
    tinfo
    ${TORCH_LIBRARIES}
)


add_definitions(-DNDEBUG)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall -Wno-unused-result -O3 -g")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall -O0 -Wno-unused-result -g")


add_executable (
 Hex
 ${source_files}
)

target_link_libraries(Hex
    ${externLibs}
)



