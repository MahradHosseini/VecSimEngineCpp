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

int TestBgeEmbedderONNXRuntime(const std::string &onnxFile, const std::string &tokenizerFile) {
    try {
        BgeTokenizerSentencePiece tokenizer(tokenizerFile);
        BgeEmbedderONNXRuntime embedder(onnxFile, 1, 1);
        const std::vector<std::string> texts = {
            "ChatGPT is amazing.",
            "OpenAI builds artificial intelligence."
        };
        BgeTokenizerSentencePiece::Encoded encoded = tokenizer.encode(texts, true, true);

        std::vector<std::vector<float>> emb1 = embedder.run(encoded);
        std::vector<std::vector<float>> emb2 = embedder.run(encoded);

        assert(emb1.size() == texts.size() && "Batch size mismatch");
        assert(!emb1.empty() && !emb2.empty() && "Empty output");

        const std::size_t hid = emb1.front().size();
        assert(hid > 0 && "Hidden size is zero");

        for (const std::vector<float> &vec : emb1) {
            assert(vec.size() == hid && "Inconsistent hidden size");
            for (float v : vec) {
                assert(std::isfinite(v) && "NaN/Inf in embedding");
            }
        }

        constexpr float eps = 1e-5f;
        for (std::size_t b = 0; b < emb1.size(); ++b)
            for (std::size_t h = 0; h < hid; ++h)
                assert(std::fabs(emb1[b][h] - emb2[b][h]) < eps &&
                       "Non-deterministic output for identical input");

        std::cout << "[OK] BgeEmbedderONNXRuntime test passed.\n";
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "[FAIL] Exception: " << e.what() << '\n';
        return 1;
    }
}
