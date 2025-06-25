//
// Created by Mahrad Hosseini on 25.06.2025.
//

#pragma once
#ifndef VECTORSIMILARITYENGINE_H
#define VECTORSIMILARITYENGINE_H

#include <vector>
#include "BgeTokenizerSentencePiece.h"
#include <onnxruntime_cxx_api.h>

class VectorSimilarityEngine {
public:
    // `skillPool`: the list of known skills must match the order of embeddings in SKILL_POOL.
    // `tokenizer`: shared tokenizer instance (HuggingFace tokenizers‑cpp, sentencepiece, etc.).
    // `embedder`: ONNX Runtime session loaded with a feature‑extraction model that returns "last_hidden_state".
    // `epsilon`: small value to avoid division by zero.
    VectorSimilarityEngine(
        // TODO: Tokenizer and Embedder should be internally initialized
        // TODO: skillPool should be passed by ref
        // TODO: remove epsilon
        // TODO: remove meanPool()
        std::vector<std::string> skillPool,
        std::shared_ptr<BgeTokenizerSentencePiece> tokenizer,
        std::shared_ptr<Ort::Session> embedder,
        float epsilon = 1e-9f
    );

    // TODO: make the return type a data structure
    [[nodiscard]] std::pair<std::vector<std::string>, std::vector<float>> getTopSkills(
        const std::string &chat, std::size_t k = 5) const;

private:
    static float dotProduct(const std::vector<float> &a, const std::vector<float> &b);

    static float l2Norm(const std::vector<float> &v);

    // Multiple texts embedder
    [[nodiscard]] std::vector<std::vector<float> > getEmbeddings(const std::vector<std::string> &texts) const;

    // Single text embedder
    [[nodiscard]] std::vector<float> getEmbedding(const std::string &text) const;

    // Members
    std::shared_ptr<BgeTokenizerSentencePiece> tokenizer_;
    std::shared_ptr<Ort::Session> embedder_;
    std::vector<std::string> skillPool_;
    std::vector<std::vector<float> > skillsEmbeddings_;
    std::vector<float> skillNorms_;
    float epsilon_;
};

#endif //VECTORSIMILARITYENGINE_H
