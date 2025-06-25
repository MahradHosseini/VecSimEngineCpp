#include "Tests.h"

int main(int argc, char **argv) {
    const std::string modelFile{"/Users/payedapay/GitHub/VecSimEngineCpp/sentencepiece.bpe.model"};
    const int result = TestBgeTokenizerSentencePiece(modelFile, 128, 100);
    return result;
}
