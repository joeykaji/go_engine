#include "mcts.hpp"
#include <iostream>

MCTS::MCTS(double explorationConstant, float komi, const std::string& modelPath)
    : C(explorationConstant), komi(komi) {
    model = torch::jit::load(modelPath);
    model.eval();
}

std::pair<std::vector<float>, float> MCTS::evalPosition(const boardstate& state) {
    auto tensor = torch::zeros({1, 3, 19, 19});
    auto acc = tensor.accessor<float, 4>();
    for (int r = 0; r < 19; r++) {
        for (int c = 0; c < 19; c++) {
            if (state.blackMove ? state.black.getStone(r,c) : state.white.getStone(r,c))
                acc[0][0][r][c] = 1.0f;
            if (state.blackMove ? state.white.getStone(r,c) : state.black.getStone(r,c))
                acc[0][1][r][c] = 1.0f;
            acc[0][2][r][c] = state.blackMove ? 1.0f : 0.0f;
        }
    }
    auto output = model.forward({tensor}).toTuple();
    auto policy = torch::softmax(output->elements()[0].toTensor(), 1)[0];
    float value = output->elements()[1].toTensor().item<float>();
    std::vector<float> policyVec(362);
    for (int i = 0; i < 362; i++)
        policyVec[i] = policy[i].item<float>();
    return {policyVec, value};
}

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

MCTS::MCTSNode* MCTS::bestPUCTChild(MCTSNode* node) {
    MCTSNode* best = nullptr;
    double bestScore = -1e9;
    for (auto& child : node->children) {
        double exploitation = child->wins / (child->visits + 1e-9);
        double exploration  = C * child->prior * std::sqrt(node->visits) / (1 + child->visits);
        double score = exploitation + exploration;
        if (score > bestScore) { bestScore = score; best = child.get(); }
    }
    return best;
}

void MCTS::expand(MCTSNode* node, const boardstate& state,
                  const std::unordered_set<uint64_t>& history,
                  const std::vector<float>& policy) {
    if (node->expanded) return;
    node->expanded = true;
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
    node->children.push_back(
        std::make_unique<MCTSNode>(-1, node, policy[361]));
}

void MCTS::backpropagate(MCTSNode* node, double result) {
    while (node != nullptr) {
        node->visits++;
        node->wins += result;
        result = 1.0 - result;
        node = node->parent;
    }
}

int MCTS::getBestMove(const boardstate& root,
                      const std::unordered_set<uint64_t>& history,
                      int iterations) {
    auto rootNode = std::make_unique<MCTSNode>(-1, nullptr, 0.0f);

    for (int i = 0; i < iterations; ++i) {
        boardstate leafState = root;
        std::unordered_set<uint64_t> leafHistory = history;
        MCTSNode* leaf = select(rootNode.get(), root, history, leafState, leafHistory);

        auto [policy, value] = evalPosition(leafState);
        expand(leaf, leafState, leafHistory, policy);

        double result = (value + 1.0f) / 2.0f;
        backpropagate(leaf, result);
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
