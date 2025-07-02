#include "Tests.h"

int main(int argc, char **argv) {
    const std::string tokenizerFile{"/Users/payedapay/GitHub/VecSimEngineCpp/sentencepiece.bpe.model"};
    const std::string onnxFile{"/Users/payedapay/bge-m3-int8/model_quantized.onnx"};
    const std::string chatsFile{"/Users/payedapay/GitHub/VecSimEngineCpp/chats.jsonl"};
    // const int result = TestBgeTokenizerSentencePiece(tokenizerFile, 128, 100);
    // const int result = TestBgeEmbedderONNXRuntime(onnxFile, tokenizerFile, 1000);
    const int result = TestVectorSimilarityEngine(tokenizerFile, onnxFile, chatsFile);

    return result;
}
