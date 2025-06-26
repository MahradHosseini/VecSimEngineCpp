//
// Created by Mahrad Hosseini on 25.06.2025.
//

#include <chrono>
#include <iostream>
#include <regex>

#include "Tests.h"

#include <fstream>

#include "BgeEmbedderONNXRuntime.h"
#include "BgeTokenizerSentencePiece.h"
#include "VectorSimilarityEngine.h"

// Testing BgeTokenizerSentencePiece
int TestBgeTokenizerSentencePiece(
    const std::string &modelFile,
    const std::size_t maxSeqLen,
    const std::size_t N
) {
    // Testing BgeTokenizerSentencePiece
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<std::chrono::steady_clock> t0 = clock::now();

    try {
        const BgeTokenizerSentencePiece tokenizer(modelFile, maxSeqLen);
        const std::vector<std::string> baseTexts = {
            "General Database Issues",
            "AI System Related Issues",
            "Operating System Related Issues",
            "Application Software Related Issues",
            "Network Issues",
            "Hardware Malfunctions",
            "Billing Issues",
            "Payment Issues",
            "Subscription Issues",
            "Staff Issues",
            "Legal Issues"
        };
        std::vector<std::string> texts;
        texts.reserve(baseTexts.size() * N);
        for (std::size_t i = 0; i < N; ++i) {
            texts.insert(texts.end(), baseTexts.begin(), baseTexts.end());
        }
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
    std::chrono::time_point<std::chrono::steady_clock> t1 = clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Time Elapsed: " << ms << " ms\n";

    return 0;
}

// ----------------------------------------------------------------------------------------------------------------

int TestBgeEmbedderONNXRuntime(
    const std::string &onnxFile,
    const std::string &tokenizerFile,
    const std::size_t N
) {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<std::chrono::steady_clock> t0 = clock::now();

    try {
        BgeTokenizerSentencePiece tokenizer(tokenizerFile);
        BgeEmbedderONNXRuntime embedder(onnxFile, 1, 1);
        const std::vector<std::string> baseTexts = {
            "General Database Issues",
            "AI System Related Issues",
            "Operating System Related Issues",
            "Application Software Related Issues",
            "Network Issues",
            "Hardware Malfunctions",
            "Billing Issues",
            "Payment Issues",
            "Subscription Issues",
            "Staff Issues",
            "Legal Issues"
        };
        std::vector<std::string> texts;
        texts.reserve(baseTexts.size() * N);
        for (std::size_t i = 0; i < N; ++i) {
            texts.insert(texts.end(), baseTexts.begin(), baseTexts.end());
        }
        BgeTokenizerSentencePiece::Encoded encoded = tokenizer.encode(texts, true, true);

        std::vector<std::vector<float> > emb1 = embedder.run(encoded);

        std::cout << "Tokenizer Shape: [" << encoded.shape[0] << ", " << encoded.shape[1] << "]\n";
        std::cout << "Embedded Vector Size: " << emb1.size() << std::endl;

        /*
        for (std::vector<float>& v : emb1) {
            std::cout << "[";
            for (float i : v) {
                std::cout << i << " ";
            }
            std::cout << "]\n";
        }
        */
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    std::chrono::time_point<std::chrono::steady_clock> t1 = clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Time Elapsed: " << ms << " ms\n";
    return 0;
}

// ----------------------------------------------------------------------------------------------------------------

static Chat parseChatLine(const std::string &line) {
    Chat chat;

    static const std::regex messageRe(R"MSG(\{"role"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]*)"\})MSG");

    for (std::sregex_iterator it = std::sregex_iterator(line.begin(), line.end(), messageRe);
         it != std::sregex_iterator(); ++it) {
        chat.messages.push_back({(*it)[1].str(), (*it)[2].str()});
    }

    static const std::regex skillsBlockRe(R"SKILLS("skills"\s*:\s*\[([^\]]*)\])SKILLS");

    std::smatch skillsMatch;
    if (std::regex_search(line, skillsMatch, skillsBlockRe)) {
        std::string inner = skillsMatch[1].str();
        static const std::regex skillRe(R"SKILL("([^\"]+)")SKILL");
        for (std::sregex_iterator it = std::sregex_iterator(inner.begin(), inner.end(), skillRe);
             it != std::sregex_iterator(); ++it) {
            chat.skills.push_back((*it)[1].str());
        }
    }
    return chat;
}

void printChats(const std::vector<Chat> &chats) {
    std::cout << "Loaded " << chats.size() << " chat(s).\n\n";
    for (std::size_t i = 0; i < chats.size(); ++i) {
        const Chat &c = chats[i];
        std::cout << "Chat #" << (i + 1) << std::endl;
        std::cout << " Messages: " << c.messages.size() << std::endl;
        for (const Message &m: c.messages) {
            std::cout << "   [" << m.role << "] " << m.text << std::endl;
        }
        std::cout << "  Skills: ";
        for (std::size_t k = 0; k < c.skills.size(); ++k) {
            std::cout << c.skills[k] << (k + 1 == c.skills.size() ? "" : ", ");
        }
        std::cout << std::endl << std::endl;
    }
}

int TestVectorSimilarityEngine(
    const std::string &tokenizerFile,
    const std::string &embedderFile,
    const std::string &chatsFile
) {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<std::chrono::steady_clock> t0 = clock::now();

    const std::vector<std::string> skillPool = {
        "General Database Issues",
        "AI System Related Issues",
        "Operating System Related Issues",
        "Application Software Related Issues",
        "Network Issues",
        "Hardware Malfunctions",
        "Billing Issues",
        "Payment Issues",
        "Subscription Issues",
        "Staff Issues",
        "Legal Issues"
    };

    std::ifstream file(chatsFile);
    if (!file) {
        std::cerr << "Cannot open " << chatsFile << '\n';
        return 1;
    }

    std::string line;
    std::vector<Chat> chats;
    std::size_t lineNo = 0;

    while (std::getline(file, line)) {
        ++lineNo;
        try {
            chats.push_back(parseChatLine(line));
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << '\n';
            return 1;
        }
    }

    // printChats(chats);

    std::size_t skillSize = skillPool.size();
    std::vector<int64_t> hits(skillSize, 0);
    int64_t total{0};

    try {
        VectorSimilarityEngine engine(skillPool, tokenizerFile, embedderFile);

        for (std::size_t i = 0; i < chats.size(); ++i) {
            std::cout << "\r" << "Processing chat " << (i + 1) << "/" << chats.size() << std::flush;
            Chat &c = chats[i];
            total += c.skills.size();
            std::string cStr;
            for (Message &m: c.messages) {
                cStr += "\n" + m.text;
            }
            // std::cout << "   Chat string: " << cStr << std::endl;
            VectorSimilarityEngine::SkillAndScoreVector tops = engine.getTopSkills(cStr, skillSize);
            for (std::size_t k = 0; k < tops.size(); ++k) {
                for (std::string &s: c.skills) {
                    if (tops[k].first == s) {
                        // std::cout << "      Hit found at " << k << " for: " << tops[k].first << std::endl;
                        hits[k] += 1;
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Total Skills: " << total << std::endl;
    int64_t totalHits{0};
    for (int64_t k: hits) {
        totalHits += k;
    }
    std::cout << "Total hits: " << totalHits << std::endl;
    for (std::size_t i = 0; i < skillSize; ++i) {
        std::cout << "   Hit Rate @" << (i + 1) << ": " << static_cast<double>(hits[i]) * 100 / total << "%" <<
                std::endl;
    }

    std::cout << std::endl;
    std::cout << "Cumulative Accuracy: " << std::endl;
    int64_t cumHits{0};
    for (std::size_t i = 0; i < skillSize; ++i) {
        cumHits += hits[i];
        std::cout << "   @" << (i + 1) << ": " << static_cast<double>(cumHits) * 100 / total << "%" << std::endl;
    }


    std::chrono::time_point<std::chrono::steady_clock> t1 = clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Time Elapsed: " << ms << " ms\n";
    return 0;
}
