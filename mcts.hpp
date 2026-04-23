#ifndef MCTS_HPP
#define MCTS_HPP
#include <vector>
#include <memory>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <random>
#include "boardstate.hpp"
#include "network.hpp"

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

    MCTS(const std::string& modelPath,
         double explorationConstant = 1.41,
         float komi = 6.5f);

    int getBestMove(const boardstate& root,
                    const std::unordered_set<uint64_t>& history,
                    int iterations);

private:
    double C;
    float komi;
    GoNetwork net;  // neural network

    MCTSNode* select(MCTSNode* node,
                     const boardstate& rootState,
                     const std::unordered_set<uint64_t>& rootHistory,
                     boardstate& outState,
                     std::unordered_set<uint64_t>& outHistory);

    // now uses policy priors from network
    MCTSNode* bestPUCTChild(MCTSNode* node);

    // now sets prior probabilities from network policy head
    double expand(MCTSNode* node, const boardstate& state,
                const std::unordered_set<uint64_t>& history);

    // replaced by network value head
    double evaluate(const boardstate& state);

    void backpropagate(MCTSNode* node, double result);
    std::mt19937 rng{42};
};

#endif
