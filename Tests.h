//
// Created by Mahrad Hosseini on 25.06.2025.
//

#pragma once
#ifndef TESTS_H
#define TESTS_H
#include <string>

int TestBgeTokenizerSentencePiece(
    const std::string &modelFile,
    std::size_t maxSeqLen = 512,
    std::size_t N = 100
    );

int TestBgeEmbedderONNXRuntime(
    const std::string &onnxFile,
    const std::string &tokenizerFile,
    std::size_t N = 10
    );

#endif //TESTS_H
