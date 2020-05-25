cmake_minimum_required(VERSION 3.0.0)
project(uzawa VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

add_executable(uzawa main.cpp gauss.cpp uzawa.cpp)
add_executable(uzawa_mpi main_mpi.cpp gauss.cpp uzawa_mpi.cpp)
target_link_libraries(uzawa PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(uzawa_mpi PUBLIC ${Boost_LIBRARIES})
target_link_libraries(uzawa_mpi PUBLIC MPI::MPI_CXX)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
