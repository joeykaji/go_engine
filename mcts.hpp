#ifndef MCTS_HPP
#define MCTS_HPP
#include <vector>
#include <memory>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include "boardstate.hpp"
#include <torch/script.h>


class MCTS {
public:
    struct MCTSNode {
        int pos;
        int visits = 0;
        double wins = 0.0;
        float prior = 0.0f;   // policy probability from network
        bool expanded = false;
        MCTSNode* parent = nullptr;
        std::vector<std::unique_ptr<MCTSNode>> children;
        MCTSNode(int pos, MCTSNode* parent, float prior = 0.0f)
          : pos(pos), prior(prior), parent(parent) {}
    };

    MCTS(double explorationConstant = 1.41,
         float komi = 6.5f,
         const std::string& modelPath = "gonet.pt");

    int getBestMove(const boardstate& root,
                    const std::unordered_set<uint64_t>& history,
                    int iterations);

private:
    double C;
    float komi;
    torch::jit::script::Module model;
    std::mt19937 rng{42};
    std::pair<std::vector<float>, float> evalPosition(const boardstate& state);

    MCTSNode* select(MCTSNode* node,
                     const boardstate& rootState,
                     const std::unordered_set<uint64_t>& rootHistory,
                     boardstate& outState,
                     std::unordered_set<uint64_t>& outHistory);

    MCTSNode* bestPUCTChild(MCTSNode* node);
    void      expand(MCTSNode* node, const boardstate& state,
                     const std::unordered_set<uint64_t>& history,
                     const std::vector<float>& policy);
    void      backpropagate(MCTSNode* node, double result);
};

#endif
