//
// Created by Mahrad Hosseini on 25.06.2025.
//

# pragma once
#ifndef BGETOKENIZERSENTENCEPIECE_H
#define BGETOKENIZERSENTENCEPIECE_H

# include <sentencepiece_processor.h>

class BgeTokenizerSentencePiece {
public:
    struct Encoded {
        std::vector<int64_t> input_ids; // Flattened [batch, seq]
        std::vector<int64_t> attention_mask; // Flattened [batch, seq]
        std::vector<int64_t> shape; // [batch, seq]
    };

    struct SpecialTokens {
        std::string pad{"<pad>"};
        std::string unk{"<unk>"};
        std::string bos{"<s>"}; // Beginning‑of‑sentence token according to BGE
        std::string eos{"</s>"}; // End‑of‑sentence token
    };

    explicit BgeTokenizerSentencePiece(
        const std::string &modelFile,
        std::size_t maxSeqLen = 512
    );


    [[nodiscard]] Encoded encode(const std::vector<std::string> &texts,
                                 bool padding = true,
                                 bool truncation = true) const;

private:
    std::size_t maxSeqLen_{512};
    SpecialTokens specials_{};

    sentencepiece::SentencePieceProcessor spm_{};

    // Special tokens' IDs based on BGE-M3 config.json at HF
    int64_t padId_{1};
    int64_t unkId_{3};
    int64_t bosId_{0};
    int64_t eosId_{2};

    // Every non-special token pluses with <offset_> to match the HF implementation.
    int64_t offset_{1};
};

#endif //BGETOKENIZERSENTENCEPIECE_H
