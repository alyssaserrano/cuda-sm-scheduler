cmake_minimum_required(VERSION 3.18)
project(cuda_sm_scheduler LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA explicitly
find_package(CUDA REQUIRED)

# Import pre-built libsmctrl.a
add_library(smctrl STATIC IMPORTED)
set_target_properties(smctrl PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libsmctrl/libsmctrl.a
)

# Add executable
add_executable(run_scheduler src/parallel_kernels_test.cu)

# Include directories
target_include_directories(run_scheduler PRIVATE ${CMAKE_SOURCE_DIR}/lib/libsmctrl)

# Link libraries manually
target_link_libraries(run_scheduler PRIVATE smctrl ${CUDA_CUDA_LIBRARY} pthread)

