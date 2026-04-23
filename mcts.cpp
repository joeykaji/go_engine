#include "mcts.hpp"
#include <random>
#include <iostream>

MCTS::MCTS(const std::string& modelPath, double explorationConstant, float komi)
    : C(explorationConstant), komi(komi), net(modelPath) {}

// ── select ────────────────────────────────────────────────────────────────────
MCTS::MCTSNode* MCTS::select(MCTSNode* node,
                              const boardstate& rootState,
                              const std::unordered_set<uint64_t>& rootHistory,
                              boardstate& outState,
                              std::unordered_set<uint64_t>& outHistory) {
    while (node->expanded && !node->children.empty()) {
        node = bestPUCTChild(node);
        outHistory.insert(outState.zobristHash);
        if (node->pos == -1) outState = outState.makePass();
        else outState = outState.makeMove(node->pos);
    }
    return node;
}

// ── bestPUCTChild ─────────────────────────────────────────────────────────────
MCTS::MCTSNode* MCTS::bestPUCTChild(MCTSNode* node) {
    MCTSNode* best = nullptr;
    double bestScore = -1e9;
    for (auto& child : node->children) {
        double exploitation = child->wins / (child->visits + 1e-9);
        double exploration  = C * child->prior
                            * std::sqrt(node->visits + 1)
                            / (child->visits + 1);
        double score = exploitation + exploration;
        if (score > bestScore) {
            bestScore = score;
            best = child.get();
        }
    }
    return best;
}

// ── expand ────────────────────────────────────────────────────────────────────
double MCTS::expand(MCTSNode* node, const boardstate& state,
                  const std::unordered_set<uint64_t>& history) {
    if (node->expanded) return 0.0;
    node->expanded = true;

    // get policy from network
    std::array<float, 361> black{}, white{};
    for (int i = 0; i < 361; i++) {
        black[i] = (state.black.w[i/64] >> (i%64)) & 1;
        white[i] = (state.white.w[i/64] >> (i%64)) & 1;
    }
    auto [policy, value] = net.inference(black, white, state.blackMove);

    bitboard legal = state.legalMoves(history);
    for (int i = 0; i < 6; ++i) {
        uint64_t word = legal.w[i];
        while (word) {
            int bit = __builtin_ctzll(word);
            int pos = i * 64 + bit;
            if (pos < 361)
                node->children.push_back(
                    std::make_unique<MCTSNode>(pos, node, policy[pos]));
            word &= word - 1;
        }
    }
    // pass move uses last policy slot (index 361)
    node->children.push_back(
        std::make_unique<MCTSNode>(-1, node, policy[361]));
    return (value + 1.0) / 2.0;
}

// ── evaluate ──────────────────────────────────────────────────────────────────
double MCTS::evaluate(const boardstate& state) {
    std::array<float, 361> black{}, white{};
    for (int i = 0; i < 361; i++) {
        black[i] = (state.black.w[i/64] >> (i%64)) & 1;
        white[i] = (state.white.w[i/64] >> (i%64)) & 1;
    }
    auto [policy, value] = net.inference(black, white, state.blackMove);
    // value is in [-1, 1], convert to [0, 1] for backprop
    return (value + 1.0) / 2.0;
}

// ── backpropagate ─────────────────────────────────────────────────────────────
void MCTS::backpropagate(MCTSNode* node, double result) {
    while (node != nullptr) {
        node->visits++;
        node->wins += result;
        result = 1.0 - result;
        node = node->parent;
    }
}

// ── getBestMove ───────────────────────────────────────────────────────────────
int MCTS::getBestMove(const boardstate& root,
                      const std::unordered_set<uint64_t>& history,
                      int iterations) {
    auto rootNode = std::make_unique<MCTSNode>(-1, nullptr);

    for (int i = 0; i < iterations; ++i) {
        boardstate leafState = root;
        std::unordered_set<uint64_t> leafHistory = history;

        MCTSNode* leaf = select(rootNode.get(), root, history,
                                leafState, leafHistory);
        expand(leaf, leafState, leafHistory);

        MCTSNode* child = leaf;
        if (!leaf->children.empty()) {
            int idx = rng() % leaf->children.size();
            child = leaf->children[idx].get();
            if (child->pos == -1) leafState = leafState.makePass();
            else leafState = leafState.makeMove(child->pos);
        }

        double result = expand(leaf, leafState, leafHistory);
        backpropagate(child, result);
    }

    MCTSNode* best = nullptr;
    int bestVisits = -1;
    for (auto& child : rootNode->children) {
        if (child->visits > bestVisits) {
            bestVisits = child->visits;
            best = child.get();
        }
    }

    std::cout << "Total children: " << rootNode->children.size() << "\n";
    std::cout << "Best pos: " << (best ? best->pos : -1)
              << " visits: " << bestVisits << "\n";

    return best ? best->pos : -1;
}
