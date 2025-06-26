//
// Created by Mahrad Hosseini on 25.06.2025.

#include <cassert>
#include<numeric>
#include "VectorSimilarityEngine.h"

VectorSimilarityEngine::VectorSimilarityEngine(
    const std::vector<std::string>& skillPool,
    const std::string &tokenizerFilePath,
    const std::string &embedderFilePath
): tokenizer_(std::make_shared<BgeTokenizerSentencePiece>(tokenizerFilePath, 512)),
   embedder_(std::make_shared<BgeEmbedderONNXRuntime>(embedderFilePath, 1, 1)),
   skillPool_(skillPool) {
    skillsEmbeddings_ = getEmbeddings(skillPool_);
    skillNorms_.reserve(skillsEmbeddings_.size());
    for (const std::vector<float> &row: skillsEmbeddings_) {
        skillNorms_.push_back(l2Norm(row));
    }
}

VectorSimilarityEngine::SkillAndScoreVector VectorSimilarityEngine::getTopSkills(
    const std::string &chat, const std::size_t k) const {
    const std::vector<float> chatVec = getEmbedding(chat);
    const float chatNorm = l2Norm(chatVec);

    // Cosine Similarity against every skill
    const std::size_t numSkills = skillsEmbeddings_.size();
    std::vector<float> sims(numSkills);
    for (std::size_t i = 0; i < numSkills; ++i) {
        float dot = dotProduct(skillsEmbeddings_[i], chatVec);
        sims[i] = dot / ((skillNorms_[i] * chatNorm) + epsilon_);
    }

    std::vector<std::size_t> idx(numSkills);
    std::iota(idx.begin(), idx.end(), 0);
    if (k < numSkills) {
        std::ranges::partial_sort(idx, idx.begin() + k, [&sims](std::size_t a, std::size_t b) {
            return sims[a] > sims[b];
        });
        idx.resize(k);
    } else {
        std::ranges::sort(idx, [&sims](std::size_t a, std::size_t b) {
            return sims[a] > sims[b];
        });
    }

    std::vector<std::string> topSkills;
    std::vector<float> topScores;
    SkillAndScoreVector tops;
    for (std::size_t i: idx) {
        tops.emplace_back(skillPool_[i], sims[i]);
    }
    return tops;
}

float VectorSimilarityEngine::dotProduct(const std::vector<float> &a, const std::vector<float> &b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

float VectorSimilarityEngine::l2Norm(const std::vector<float> &v) {
    return std::sqrt(dotProduct(v, v));
}

std::vector<std::vector<float> > VectorSimilarityEngine::getEmbeddings(const std::vector<std::string> &texts) const {
    // Tokenize
    BgeTokenizerSentencePiece::Encoded enc = tokenizer_->encode(texts, true, true);

    // Embedd
    std::vector<std::vector<float>> emb = embedder_->run(enc);

    return emb;
}

std::vector<float> VectorSimilarityEngine::getEmbedding(const std::string &text) const {
    std::vector<std::vector<float> > res = getEmbeddings({text});
    return res.front();
}