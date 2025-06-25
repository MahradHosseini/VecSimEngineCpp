//
// Created by Mahrad Hosseini on 25.06.2025.
//

#pragma once
#ifndef BGEEMBEDDERONNXRUNTIME_H
#define BGEEMBEDDERONNXRUNTIME_H

#include<vector>
#include <onnxruntime_cxx_api.h>
#include "BgeTokenizerSentencePiece.h"

class BgeEmbedderONNXRuntime {
public:
    BgeEmbedderONNXRuntime(
        const std::string &modelPath,
        int intraThreads,
        int interThreads
    );
    [[nodiscard]] std::vector<std::vector<float>> run(const BgeTokenizerSentencePiece::Encoded &encoded) const;

private:
    std::vector<std::vector<float> > meanPool(
        const float *lastHiddenState,
        int64_t batch,
        int64_t seq,
        int64_t hid,
        const std::vector<int64_t> &attention_mask
    ) const;

    const float epsilon_{1e-9f};
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> embedder_;
};

#endif //BGEEMBEDDERONNXRUNTIME_H
