#include "Tests.h"

int main(int argc, char **argv) {
    const std::string tokenizerFile{"/Users/payedapay/GitHub/VecSimEngineCpp/sentencepiece.bpe.model"};
    const std::string onnxFile{"/Users/payedapay/GitHub/VecSimEngineCpp/model_quantized.onnx"};
    const int result = TestBgeTokenizerSentencePiece(tokenizerFile, 128, 100);

    return result;
}
