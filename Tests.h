//
// Created by Mahrad Hosseini on 25.06.2025.
//

#pragma once
#ifndef TESTS_H
#define TESTS_H
#include <string>
#include <vector>


struct Message {
    std::string role;
    std::string text;
};

struct Chat {
    std::vector<Message> messages;
    std::vector<std::string> skills;
};

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

int TestVectorSimilarityEngine(
    const std::string &tokenizerFile,
    const std::string &embedderFile,
    const std::string &chatsFile
    );

void printChats(const std::vector<std::string> &chats);
static Chat parseChatLine(const std::string &line);

#endif //TESTS_H
