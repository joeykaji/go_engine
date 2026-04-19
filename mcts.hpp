#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include "boardstate.hpp"


class MCTS {
public:
    struct MCTSNode {
    boardstate state;
    int pos;
    int visits = 0;
    double wins = 0.0;
    bool expanded = false;
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode(const boardstate& state, int pos, MCTSNode* parent)
        : state(state), pos(pos), parent(parent) {}
    };

    MCTS(double explorationConstant = 1.41, float komi = 6.5f);

    int getBestMove(const boardstate& root,
                    const std::unordered_set<uint64_t>& history,
                    int iterations);

private:
    double C; // exploration constant
    float komi;
    MCTSNode* select(MCTSNode* node);
    MCTSNode* bestUCB1Child(MCTSNode* node);
    void      expand(MCTSNode* node, const std::unordered_set<uint64_t>& history);
    double    simulate(boardstate state);
    void      backpropagate(MCTSNode* node, double result);
};

#endif
