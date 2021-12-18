# CMakeList.txt : fichier projet CMake de niveau supérieur, effectuez une configuration globale
# et incluez les sous-projets ici.
#
cmake_minimum_required (VERSION 3.8)
set( CMAKE_TOOLCHAIN_FILE "G:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" )
project ("Maze_VS21")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Set OpenCV Config.cmake Search directory
set( OpenCV_DIR )

# Find opencv package
find_package(OpenCV REQUIRED)

find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp)

include_directories(${YamlCpp_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

# Incluez les sous-projets.

add_library(Maze_VS21 SHARED include/maze.h include/a_star.h
    include/point.h src/point.cpp)
target_link_libraries(Maze_VS21 ${OpenCV_LIBS})

#add_executable(generator src/generator.cpp)
#target_link_libraries(generator maze)

#add_executable(line src/solve_line.cpp)
#target_link_libraries(line maze)

add_executable(cell src/solve_cell.cpp)
target_link_libraries(cell  maze)

#add_executable(corridor src/solve_corridor.cpp)
#target_link_libraries(corridor maze)

