//
// Created by Mahrad Hosseini on 25.06.2025.
//

#pragma once
#ifndef VECTORSIMILARITYENGINE_H
#define VECTORSIMILARITYENGINE_H

#include <vector>
#include <onnxruntime_cxx_api.h>
#include "BgeTokenizerSentencePiece.h"
#include "BgeEmbedderONNXRuntime.h"

class VectorSimilarityEngine {
public:
    typedef std::pair<std::string, float> SkillAndScore;
    typedef std::vector<SkillAndScore> SkillAndScoreVector;

    VectorSimilarityEngine(
        const std::vector<std::string>& skillPool,
        const std::string &tokenizerFilePath,
        const std::string &embedderFilePath
        );

    [[nodiscard]] SkillAndScoreVector getTopSkills(
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
    std::shared_ptr<BgeEmbedderONNXRuntime> embedder_;
    std::vector<std::string> skillPool_;
    std::vector<std::vector<float> > skillsEmbeddings_;
    std::vector<float> skillNorms_;
    const float epsilon_{1e-9f};
};

#endif //VECTORSIMILARITYENGINE_H
