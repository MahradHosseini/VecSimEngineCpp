//
// Created by Mahrad Hosseini on 25.06.2025.
//

#include <chrono>
#include <iostream>
#include <cassert>
#include "Tests.h"

#include "BgeEmbedderONNXRuntime.h"
#include "BgeTokenizerSentencePiece.h"

// Testing BgeTokenizerSentencePiece
int TestBgeTokenizerSentencePiece(
    const std::string &modelFile,
    const std::size_t maxSeqLen,
    const std::size_t N
) {
    // Testing BgeTokenizerSentencePiece
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<std::chrono::steady_clock> t0 = clock::now();

    try {
        const BgeTokenizerSentencePiece tokenizer(modelFile, maxSeqLen);
        const std::vector<std::string> baseTexts = {
            "General Database Issues",
            "AI System Related Issues",
            "Operating System Related Issues",
            "Application Software Related Issues",
            "Network Issues",
            "Hardware Malfunctions",
            "Billing Issues",
            "Payment Issues",
            "Subscription Issues",
            "Staff Issues",
            "Legal Issues"
        };
        std::vector<std::string> texts;
        texts.reserve(baseTexts.size() * N);
        for (std::size_t i = 0; i < N; ++i) {
            texts.insert(texts.end(), baseTexts.begin(), baseTexts.end());
        }
        BgeTokenizerSentencePiece::Encoded enc = tokenizer.encode(texts);
        std::cout << "Shape: [" << enc.shape[0] << ", " << enc.shape[1] << "]\n";

        // Pretty print
        auto printTensor = [&](const std::vector<int64_t> &data, const char *name) {
            std::cout << name << ": \n";
            for (std::size_t row = 0; row < enc.shape[0]; ++row) {
                for (std::size_t col = 0; col < enc.shape[1]; ++col) {
                    std::cout << data[row * enc.shape[1] + col] << " ";
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        };
        printTensor(enc.input_ids, "input_ids");
        printTensor(enc.attention_mask, "attention_mask");
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::chrono::time_point<std::chrono::steady_clock> t1 = clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Time Elapsed: " << ms << " ms\n";

    return 0;
}

int TestBgeEmbedderONNXRuntime(
    const std::string &onnxFile,
    const std::string &tokenizerFile,
    const std::size_t N
    ) {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<std::chrono::steady_clock> t0 = clock::now();

    try {
        BgeTokenizerSentencePiece tokenizer(tokenizerFile);
        BgeEmbedderONNXRuntime embedder(onnxFile, 1, 1);
        const std::vector<std::string> baseTexts = {
            "General Database Issues",
            "AI System Related Issues",
            "Operating System Related Issues",
            "Application Software Related Issues",
            "Network Issues",
            "Hardware Malfunctions",
            "Billing Issues",
            "Payment Issues",
            "Subscription Issues",
            "Staff Issues",
            "Legal Issues"
        };
        std::vector<std::string> texts;
        texts.reserve(baseTexts.size() * N);
        for (std::size_t i = 0; i < N; ++i) {
            texts.insert(texts.end(), baseTexts.begin(), baseTexts.end());
        }
        BgeTokenizerSentencePiece::Encoded encoded = tokenizer.encode(texts, true, true);

        std::vector<std::vector<float> > emb1 = embedder.run(encoded);

        std::cout << "Tokenizer Shape: [" << encoded.shape[0] << ", " << encoded.shape[1] << "]\n";
        std::cout << "Embedded Vector Size: " << emb1.size() << std::endl;

        /*
        for (std::vector<float>& v : emb1) {
            std::cout << "[";
            for (float i : v) {
                std::cout << i << " ";
            }
            std::cout << "]\n";
        }
        */
        std::chrono::time_point<std::chrono::steady_clock> t1 = clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Time Elapsed: " << ms << " ms\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] Exception: " << e.what() << '\n';
        return 1;
    }
}
