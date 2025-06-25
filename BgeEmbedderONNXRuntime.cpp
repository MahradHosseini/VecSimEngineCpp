//
// Created by Mahrad Hosseini on 25.06.2025.
//

#include "BgeEmbedderONNXRuntime.h"

BgeEmbedderONNXRuntime::BgeEmbedderONNXRuntime(
    const std::string &modelPath,
    int intraThreads,
    int interThreads
): env_(ORT_LOGGING_LEVEL_WARNING, "BgeEmbedderONNXRuntime") {
    sessionOptions_.SetIntraOpNumThreads(intraThreads);
    sessionOptions_.SetInterOpNumThreads(interThreads);
    sessionOptions_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    embedder_ = std::make_unique<Ort::Session>(
        env_,
        modelPath.c_str(),
        sessionOptions_
    );
}


std::vector<std::vector<float> > BgeEmbedderONNXRuntime::run(const BgeTokenizerSentencePiece::Encoded &encoded) const {
    // Prepare ONNX tensors
    const int64_t &batch = encoded.shape[0];
    const int64_t &seq = encoded.shape[1];
    const std::vector<int64_t> &input_ids = encoded.input_ids;
    const std::vector<int64_t> &attn_mask = encoded.attention_mask;

    const std::array<int64_t, 2> inputShape = {batch, seq};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputIdsTensor = Ort::Value::CreateTensor<int64_t>(
        memInfo,
        const_cast<int64_t *>(input_ids.data()),
        input_ids.size(),
        inputShape.data(),
        inputShape.size()
    );
    Ort::Value attnMaskTensor = Ort::Value::CreateTensor<int64_t>(
        memInfo,
        const_cast<int64_t *>(attn_mask.data()),
        attn_mask.size(),
        inputShape.data(),
        inputShape.size()
    );

    const std::array<const char *, 2> inputNames{"input_ids", "attn_mask"};
    const std::array<const char *, 2> outputNames{"last_hidden_state"};
    std::array<Ort::Value, 2> inputs{std::move(inputIdsTensor), std::move(attnMaskTensor)};

    std::vector<Ort::Value> outputs = embedder_->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputs.data(),
        inputNames.size(),
        outputNames.data(),
        outputNames.size()
    );

    const float *outData = outputs[0].GetTensorMutableData<float>();
    const std::vector<int64_t> &outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t hid = outShape[2];

    return meanPool(outData, batch, seq, hid, encoded.attention_mask);
}

std::vector<std::vector<float> > BgeEmbedderONNXRuntime::meanPool(
    const float *lastHiddenState,
    const int64_t batch,
    const int64_t seq,
    const int64_t hid,
    const std::vector<int64_t> &attention_mask
) const {
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
        const float denom = (maskSum > 0.0f) ? maskSum : epsilon_;
        for (float &v: pooled[b]) v /= denom;
    }
    return pooled;
}
