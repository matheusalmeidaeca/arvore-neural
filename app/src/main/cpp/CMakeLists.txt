cmake_minimum_required(VERSION 3.22.1)
project(sayotollm LANGUAGES C CXX)

# llama.cpp é baixado na hora do build (o GitHub Actions tem internet).
# Versão FIXA por commit (de 2 out 2026). Para atualizar, troque só esta linha.
set(LLAMA_COMMIT "4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe" CACHE STRING "commit do llama.cpp")

include(FetchContent)

# Tudo estático dentro de libsayotollm.so; sem OpenMP, sem rede, sem exemplos.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# Instruções do processador ARM. O app/build.gradle.kts pode trocar via -DGGML_CPU_ARM_ARCH=...
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod" CACHE STRING "")

# Mesmo no APK de depuração a parte nativa precisa ser otimizada (senão fica lentíssima).
set(CMAKE_C_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)

FetchContent_Declare(
    llama
    URL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"
)
FetchContent_MakeAvailable(llama)

add_library(sayotollm SHARED llm_jni.cpp)
target_compile_features(sayotollm PRIVATE cxx_std_17)
target_link_libraries(sayotollm PRIVATE llama android log)
