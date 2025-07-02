cmake_minimum_required(VERSION 3.28)

set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

project(VecSimEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
add_executable(VecSimEngine main.cpp
        BgeTokenizerSentencePiece.h
        BgeTokenizerSentencePiece.cpp
        VectorSimilarityEngine.h
        VectorSimilarityEngine.cpp
        Tests.cpp
        Tests.h
        BgeEmbedderONNXRuntime.h
        BgeEmbedderONNXRuntime.cpp)

target_include_directories(VecSimEngine PRIVATE /opt/homebrew/Cellar/onnxruntime/1.22.0/include/onnxruntime)
target_include_directories(VecSimEngine PRIVATE /opt/homebrew/Cellar/sentencepiece/0.2.0/include)
target_link_directories(VecSimEngine PRIVATE /opt/homebrew/Cellar/onnxruntime/1.22.0/lib)
target_link_directories(VecSimEngine PRIVATE /opt/homebrew/Cellar/sentencepiece/0.2.0/lib)
target_link_libraries(VecSimEngine PRIVATE onnxruntime)
target_link_libraries(VecSimEngine PRIVATE sentencepiece)
