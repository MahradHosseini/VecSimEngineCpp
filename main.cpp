#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <fstream>
#include <iostream>
#include <onnxruntime_cxx_api.h>

class BgeTokenizer {
public:
    struct Encoded {
        std::vector<int64_t> input_ids; // Flattened [batch, seq]
        std::vector<int64_t> attention_mask; // Flattened [batch, seq]
        std::vector<int64_t> shape; // [batch, seq]
    };

    struct SpecialTokens {
        std::string pad = "<pad>";
        std::string unk = "<unk>";
        std::string bos = "<s>"; // Beginning of the sentence
        std::string eos = "</s>"; // End of the sentence
        constexpr SpecialTokens() : pad("<pad>"), unk("<unk>"), bos("<s>"), eos("</s>") {
        }
    };

    explicit BgeTokenizer(
        const std::string &vocabFile,
        const std::size_t maxSeqLen = 512,
        SpecialTokens specials = SpecialTokens()
    ) : maxSeqLen_(maxSeqLen), specials_(std::move(specials)) {
        loadVocab(vocabFile);
    }

    [[nodiscard]] Encoded encode(const std::vector<std::string> &texts,
                                 bool padding = true,
                                 bool truncation = true
    ) const {
        const std::size_t batch = texts.size();
        std::vector<std::vector<int64_t> > batchIds;
        batchIds.reserve(batch);
        std::size_t maxLenInBatch = 0;

        // Tokenize
        for (const std::string &t: texts) {
            std::vector<int64_t> ids;
            tokeniseToIds(t, ids);

            if (truncation && ids.size() > maxSeqLen_) {
                ids.resize(maxSeqLen_);
            }
            maxLenInBatch = std::max(maxLenInBatch, ids.size());
            batchIds.push_back(std::move(ids));
        }
        const std::size_t seq = padding ? maxLenInBatch : maxSeqLen_;

        // Flatten and build masks
        Encoded enc;
        enc.shape = {static_cast<int64_t>(batch), static_cast<int64_t>(seq)};
        enc.input_ids.reserve(batch * seq);
        enc.attention_mask.reserve(batch * seq);

        for (auto &ids: batchIds) {
            // Right Padding
            if (padding && ids.size() < seq) {
                ids.resize(seq, padId_);
            }

            // Attention mask 1 for real token, else 0
            for (std::size_t i = 0; i < seq; ++i) {
                if (i < ids.size() && ids[i] != padId_) {
                    enc.attention_mask.push_back(1);
                } else {
                    enc.attention_mask.push_back(0);
                }
            }
            // Copy ids
            enc.input_ids.insert(enc.input_ids.end(), ids.begin(), ids.end());
        }
        return enc;
    }

private:
    void loadVocab(const std::string &file) {
        std::ifstream ifs(file);
        if (!ifs.good()) {
            throw std::runtime_error("BgeTokenizer: cannot open vocab file" + file);
        }
        std::string line;
        int64_t id = 0;
        while (std::getline(ifs, line)) {
            if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                line.pop_back();
            }
            idToToken_.push_back(line);
            tokenToId_[line] = id;
            ++id;
        }
        padId_ = lookupId(specials_.pad);
        unkId_ = lookupId(specials_.unk);
        bosId_ = lookupId(specials_.bos);
        eosId_ = lookupId(specials_.eos);
    }

    [[nodiscard]] int64_t lookupId(const std::string &token) const {
        auto it = tokenToId_.find(token);
        if (it == tokenToId_.end()) {
            throw std::runtime_error("Special toke '" + token + "' not found in vocab");
        }
        return it->second;
    }

    // Longest match SentencePiece style
    void tokeniseToIds(const std::string &sentence, std::vector<int64_t> &outIds) const {
        outIds.clear();
        outIds.push_back(bosId_);

        // Replace whitespace with underscore
        std::string preproc;
        preproc.reserve(sentence.size() * 2);
        bool prevSpace = true;
        for (char ch: sentence) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                prevSpace = true;
            } else {
                if (prevSpace) preproc.push_back('\xE2');
                if (prevSpace) preproc.push_back('\x96');
                if (prevSpace) preproc.push_back('\x81');
                preproc.push_back(ch);
                prevSpace = false;
            }
        }

        // Greedy left to right longest match against vocab.
        // Search window [i, j).
        std::size_t i = 0;
        const std::size_t N = preproc.size();
        while (i < N) {
            std::size_t j = N;
            int64_t matched = -1;
            while (j > i) {
                std::string_view sub(&preproc[i], j - 1);
                auto it = tokenToId_.find(std::string(sub));
                if (it != tokenToId_.end()) {
                    matched = it->second;
                    break;
                }
                --j;
            }
            if (matched == -1) {
                matched = unkId_;
                ++i;
            } else {
                i += (j - i);
            }
            outIds.push_back(matched);
            if (outIds.size() >= maxSeqLen_ - 1) break;
        }
        outIds.push_back(eosId_);
    }

    // Members
    std::size_t maxSeqLen_;
    SpecialTokens specials_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int64_t> tokenToId_;

    int64_t padId_ = 0;
    int64_t unkId_ = 0;
    int64_t bosId_ = 0;
    int64_t eosId_ = 0;
};

// --------------------------------------------------------------------------------------------

# include <sentencepiece_processor.h>

class BgeTokenizerSentencePiece {
public:
    struct Encoded {
        std::vector<int64_t> input_ids; // Flattened [batch, seq]
        std::vector<int64_t> attention_mask; // Flattened [batch, seq]
        std::vector<int64_t> shape; // [batch, seq]
    };

    struct SpecialTokens {
        std::string pad = "<pad>";
        std::string unk = "<unk>";
        std::string bos = "<s>"; // Beginning‑of‑sentence token according to BGE
        std::string eos = "</s>"; // End‑of‑sentence token
        constexpr SpecialTokens() : pad("<pad>"), unk("<unk>"), bos("<s>"), eos("</s>") {
        }
    };

    explicit BgeTokenizerSentencePiece(
        const std::string &modelFile,
        std::size_t maxSeqLen = 512,
        SpecialTokens specials = SpecialTokens()
    ) : maxSeqLen_(maxSeqLen), specials_(std::move(specials)) {
        if (const sentencepiece::util::Status status = spm_.Load(modelFile); !status.ok()) {
            throw std::runtime_error("BgeTokenizerSentencePiece: cannot load model: " + status.ToString());
        }
        // Cache special‑token ids so we don’t have to look them up repeatedly.
        /*
        padId_ = spm_.PieceToId(specials_.pad);
        unkId_ = spm_.PieceToId(specials_.unk);
        bosId_ = spm_.PieceToId(specials_.bos);
        eosId_ = spm_.PieceToId(specials_.eos);
        */

        // Special tokens' IDs based on BGE-M3 config.json at HF
        padId_ = 1;
        unkId_ = 3;
        bosId_ = 0;
        eosId_ = 2;

        // Every non-special token pluses with <offset_> to match the HF implementation.
        offset_ = 1;
    }

    [[nodiscard]] Encoded encode(const std::vector<std::string> &texts,
                                 bool padding = true,
                                 bool truncation = true) const {
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

private:
    std::size_t maxSeqLen_;
    SpecialTokens specials_;

    sentencepiece::SentencePieceProcessor spm_;

    int64_t padId_ = 0;
    int64_t unkId_ = 0;
    int64_t bosId_ = 0;
    int64_t eosId_ = 0;
    int64_t offset_ = 1;
};

// --------------------------------------------------------------------------------------------

class VectorSimilarityEngine {
public:
    // `skillPool`: the list of known skills must match the order of embeddings in SKILL_POOL.
    // `tokenizer`: shared tokenizer instance (HuggingFace tokenizers‑cpp, sentencepiece, etc.).
    // `embedder`: ONNX Runtime session loaded with a feature‑extraction model that returns "last_hidden_state".
    // `epsilon`: small value to avoid division by zero.
    VectorSimilarityEngine(
        // TODO: skillPool can be const
        std::vector<std::string> skillPool,
        std::shared_ptr<BgeTokenizer> tokenizer,
        std::shared_ptr<Ort::Session> embedder,
        float epsilon = 1e-9f
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

    // TODO: make the return type a data structure
    [[nodiscard]] std::pair<std::vector<std::string>, std::vector<float> > getTopSkills(
        const std::string &chat, std::size_t k = 5) const {
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

private:
    static float dotProduct(const std::vector<float> &a, const std::vector<float> &b) {
        assert(a.size() == b.size());
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    static float l2Norm(const std::vector<float> &v) {
        return std::sqrt(dotProduct(v, v));
    }

    // Mean‑pool hidden states using the attention mask.
    // lastHiddenState: pointer to [batch, seq, hid]
    // attention_mask: flattened [batch*seq]
    // Returns: vector< vector<float> > shaped [batch, hid]
    static std::vector<std::vector<float> > meanPool(
        const float *lastHiddenState,
        const int64_t batch,
        const int64_t seq,
        const int64_t hid,
        const std::vector<int64_t> &attention_mask,
        const float epsilon
    ) {
        std::vector<std::vector<float> > pooled(batch, std::vector<float>(hid, 0.0f));
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

    // Multiple texts embedder
    [[nodiscard]] std::vector<std::vector<float> > getEmbeddings(const std::vector<std::string> &texts) const {
        // Tokenize
        BgeTokenizer::Encoded enc = tokenizer_->encode(texts, true, true);
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


    // Single text embedder
    [[nodiscard]] std::vector<float> getEmbedding(const std::string &text) const {
        // TODO: remove auto
        auto res = getEmbeddings({text});
        return res.front();
    }

    // Members
    std::shared_ptr<BgeTokenizer> tokenizer_;
    std::shared_ptr<Ort::Session> embedder_;
    std::vector<std::string> skillPool_;
    std::vector<std::vector<float> > skillsEmbeddings_;
    std::vector<float> skillNorms_;
    float epsilon_;
};


int main(int argc, char **argv) {
    // Testing BgeTokenizerSentencePiece
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<sentencepiece_model_path> [max_seq_len]\n";
        return 1;
    }

    const std::string modelFile = argv[1];
    const size_t maxSeqLen = (argc >= 3) ? std::stoul(argv[2]) : 32; // Short for demo
    try {
        const BgeTokenizerSentencePiece tokenizer(modelFile, maxSeqLen);
        const std::vector<std::string> texts = {
            "Hello world!",
            "SentencePiece tokenization with BGE tokenizer.",
            "Short"
        };
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
    return 0;
}
