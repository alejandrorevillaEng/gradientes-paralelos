cmake_minimum_required(VERSION 3.24)
project(gradientes_paralelos LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

if(NOT MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra> $<$<COMPILE_LANGUAGE:C>:-Wpedantic>)
endif()

add_library(gradientes src/polinomio.c src/gradiente.c src/sistema.c)
target_include_directories(gradientes PUBLIC include)
target_link_libraries(gradientes PUBLIC OpenMP::OpenMP_C)
if(NOT MSVC)
  target_link_libraries(gradientes PUBLIC m)
endif()

# La misma biblioteca como .so/.dll para usarla desde Python (python/gradientes).
option(CON_PYTHON "Compilar la biblioteca compartida para Python" ON)
if(CON_PYTHON)
  add_library(gradientes_compartida SHARED src/polinomio.c src/gradiente.c src/sistema.c)
  set_target_properties(gradientes_compartida PROPERTIES OUTPUT_NAME gradientes
                        WINDOWS_EXPORT_ALL_SYMBOLS ON)
  target_include_directories(gradientes_compartida PUBLIC include)
  target_link_libraries(gradientes_compartida PUBLIC OpenMP::OpenMP_C)
  if(NOT MSVC)
    target_link_libraries(gradientes_compartida PUBLIC m)
  endif()
endif()

add_executable(gradbench src/main_bench.c)
target_link_libraries(gradbench PRIVATE gradientes)

add_executable(sistema_demo src/main_sistema.c)
target_link_libraries(sistema_demo PRIVATE gradientes)

enable_testing()
add_executable(tests_cpu tests/test_main.c)
target_link_libraries(tests_cpu PRIVATE gradientes)
add_test(NAME tests_cpu COMMAND tests_cpu)
add_test(NAME sistema_demo COMMAND sistema_demo)
add_test(NAME gradbench_humo COMMAND gradbench --vars 60 --terms 600 --reps 1 --hilos 1,2
         --tamanos 600 --experimento todo)

find_package(Python3 3.10 COMPONENTS Interpreter)
if(CON_PYTHON AND Python3_FOUND)
  add_test(NAME python COMMAND Python3::Interpreter -m unittest discover -s ${CMAKE_SOURCE_DIR}/python/tests -v)
  set_tests_properties(python PROPERTIES ENVIRONMENT "GRADIENTES_LIB=$<TARGET_FILE:gradientes_compartida>")
endif()

# CUDA es opcional: solo se compila si hay nvcc (Colab, Kaggle o un PC con NVIDIA).
option(CON_CUDA "Compilar la version CUDA si hay compilador" ON)
if(CON_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Por defecto, la arquitectura de la GPU de la máquina (T4 en Colab = 75, RTX 50xx = 120).
    # Tiene que ir antes de enable_language(CUDA), que si no pone su propio valor.
    # Sin GPU (por ejemplo en CI) hay que indicarla: -DCMAKE_CUDA_ARCHITECTURES=75
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_executable(gradbench_cuda cuda/gradiente_cuda.cu)
    target_link_libraries(gradbench_cuda PRIVATE gradientes)
    message(STATUS "CUDA encontrado: se compila gradbench_cuda")
  else()
    message(STATUS "Sin nvcc: se omite la version CUDA")
  endif()
endif()
