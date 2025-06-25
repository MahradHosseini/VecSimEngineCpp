//
// Created by Mahrad Hosseini on 25.06.2025.
//
#include "BgeTokenizerSentencePiece.h"

BgeTokenizerSentencePiece::BgeTokenizerSentencePiece(
    const std::string &modelFile,
    const std::size_t maxSeqLen
) : maxSeqLen_(maxSeqLen) {
    // TODO: download the file from HF
    if (const sentencepiece::util::Status status = spm_.Load(modelFile); !status.ok()) {
        throw std::runtime_error("BgeTokenizerSentencePiece: cannot load model: " + status.ToString());
    }
}

BgeTokenizerSentencePiece::Encoded BgeTokenizerSentencePiece::encode(
    const std::vector<std::string> &texts,
    const bool padding,
    const bool truncation) const {
    const std::size_t batch = texts.size();
    std::vector<std::vector<int64_t> > batchIds;
    batchIds.reserve(batch);

    std::size_t maxLenInBatch = 0;
    for (const std::string &t: texts) {
        std::vector<int> tmp; // SentencePiece uses int
        spm_.Encode(t, &tmp);

        // Adding offset_ to raw ids
        for (int &tok: tmp) {
            tok += static_cast<int>(offset_);
        }

        std::vector<int64_t> ids;
        ids.reserve(tmp.size() + 2);
        ids.push_back(bosId_);
        ids.insert(ids.end(), tmp.begin(), tmp.end());
        ids.push_back(eosId_);

        if (truncation && ids.size() > maxSeqLen_) {
            ids.resize(maxSeqLen_);
        }
        maxLenInBatch = std::max(maxLenInBatch, ids.size());
        batchIds.emplace_back(std::move(ids));
    }

    const std::size_t seq = padding ? maxLenInBatch : maxSeqLen_;
    Encoded enc;
    enc.shape = {static_cast<int64_t>(batch), static_cast<int64_t>(seq)};
    enc.input_ids.reserve(batch * seq);
    enc.attention_mask.reserve(batch * seq);

    for (std::vector<int64_t> &ids: batchIds) {
        if (padding && ids.size() < seq) {
            ids.resize(seq, padId_);
        }
        for (std::size_t i = 0; i < seq; ++i) {
            enc.attention_mask.push_back((i < ids.size() && ids[i] != padId_) ? 1 : 0);
        }
        enc.input_ids.insert(enc.input_ids.end(), ids.begin(), ids.end());
    }
    return enc;
}
