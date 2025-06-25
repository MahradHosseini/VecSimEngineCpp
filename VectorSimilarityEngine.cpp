//
// Created by Mahrad Hosseini on 25.06.2025.
//

#include <cassert>
#include<numeric>
#include "VectorSimilarityEngine.h"

VectorSimilarityEngine::VectorSimilarityEngine(
        std::vector<std::string> skillPool,
        std::shared_ptr<BgeTokenizerSentencePiece> tokenizer,
        std::shared_ptr<Ort::Session> embedder,
        const float epsilon
    ): tokenizer_(std::move(tokenizer)),
       embedder_(std::move(embedder)),
       skillPool_(std::move(skillPool)),
       epsilon_(epsilon) {
    skillsEmbeddings_ = getEmbeddings(skillPool_);
    skillNorms_.reserve(skillsEmbeddings_.size());
    for (const std::vector<float> &row: skillsEmbeddings_) {
        skillNorms_.push_back(l2Norm(row));
    }
}

std::pair<std::vector<std::string>, std::vector<float> > VectorSimilarityEngine::getTopSkills(
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
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&sims](std::size_t a, std::size_t b) {
            return sims[a] > sims[b];
        });
        idx.resize(k);
    }

    std::vector<std::string> topSkills;
    std::vector<float> topScores;
    topSkills.reserve(idx.size());
    topScores.reserve(idx.size());
    for (std::size_t i: idx) {
        topSkills.push_back(skillPool_[i]);
        topScores.push_back(sims[i]);
    }
    return std::make_pair(topSkills, topScores);
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

std::vector<std::vector<float> > VectorSimilarityEngine::meanPool(
    const float *lastHiddenState,
    const int64_t batch,
    const int64_t seq,
    const int64_t hid,
    const std::vector<int64_t> &attention_mask,
    const float epsilon
) {
    std::vector pooled(batch, std::vector<float>(hid, 0.0f));
    const float *base = lastHiddenState;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t bOffset = b * seq * hid;
        float maskSum = 0.0f;

        for (int64_t s = 0; s < seq; ++s) {
            const int64_t mask = attention_mask[b * seq + s];
            if (!mask) continue;
            maskSum += static_cast<float>(mask);
            const float *tokenVec = base + bOffset + s * hid;
            for (int64_t h = 0; h < hid; ++h) {
                pooled[b][h] += tokenVec[h];
            }
        }

        // Normalize
        float denom = (maskSum > 0.0f) ? maskSum : epsilon;
        for (float &v: pooled[b]) v /= denom;
    }
    return pooled;
}

std::vector<std::vector<float> > VectorSimilarityEngine::getEmbeddings(const std::vector<std::string> &texts) const {
    // Tokenize
    BgeTokenizerSentencePiece::Encoded enc = tokenizer_->encode(texts, true, true);
    assert(enc.shape.size() == 2);
    const int64_t batch = enc.shape[0];
    const int64_t seq = enc.shape[1];

    // Prepare ONNX tensors
    // TODO: check if it's needed to load memInfo every time
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> inputShape{batch, seq};
    Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(memInfo, enc.input_ids.data(),
                                                                  enc.input_ids.size(), inputShape.data(), 2);
    Ort::Value attnMaskTensor = Ort::Value::CreateTensor<int64_t>(memInfo, enc.attention_mask.data(),
                                                                  enc.attention_mask.size(), inputShape.data(), 2);

    const char *inputNames[] = {"input_ids", "attention_mask"};
    Ort::Value inputs[] = {std::move(inputIdsTensor), std::move(attnMaskTensor)};

    // Run the model
    const char *outputNames[] = {"last_hidden_state"};
    auto outputs = embedder_->Run(Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);

    // Extract raw ptr and dims
    float *outData = outputs[0].GetTensorMutableData<float>();
    const std::vector<int64_t> &outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    assert(outShape.size() == 3); // [batch, seq, hid]
    const int64_t hid = outShape[2];

    // Mean pooling
    return meanPool(outData, batch, seq, hid, enc.attention_mask, epsilon_);
}

std::vector<float> VectorSimilarityEngine::getEmbedding(const std::string &text) const {
    // TODO: remove auto
    std::vector<std::vector<float>> res = getEmbeddings({text});
    return res.front();
}