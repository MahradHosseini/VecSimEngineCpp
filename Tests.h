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

#endif //TESTS_H
