#include "mcts.hpp"
#include <limits>
#include <random>
#include <iostream>

MCTS::MCTS(double explorationConstant, float komi) : C(explorationConstant), komi(komi) {}

int MCTS::getBestMove(const boardstate& root,
                      const std::unordered_set<uint64_t>& history,
                      int iterations){
    auto rootNode = std::make_unique<MCTSNode>(root, -1, nullptr);

    for(int i = 0; i < iterations; ++i){
        MCTSNode* leaf = select(rootNode.get());
        expand(leaf, history);
        MCTSNode* child = leaf;
        if(!leaf->children.empty()){
            child = leaf->children[rand() % leaf->children.size()].get();
        }
        double result = simulate(child->state);
        backpropagate(child, result);
    }

    // pick most visited child
    MCTSNode* best = nullptr;
    int bestVisits = -1;
    for(auto& child : rootNode->children){
        if(child->visits > bestVisits){
            bestVisits = child->visits;
            best = child.get();
        }
    }
    std::cout << "Total children: " << rootNode->children.size() << "\n";
    std::cout << "Best pos: " << (best ? best->pos : -1) 
          << " visits: " << bestVisits << "\n";
    return best ? best->pos : -1; // -1 = pass
}


MCTS::MCTSNode* MCTS::select(MCTSNode* node){
    while(!node->children.empty()){
        node = bestUCB1Child(node);
    }
    return node;
}

MCTS::MCTSNode* MCTS::bestUCB1Child(MCTSNode* node){
    MCTSNode* best = nullptr;
    double bestScore = -1e9;
    for(auto& child : node->children){
        double exploitation = child->wins / (child->visits + 1e-9);
        double exploration  = std::sqrt(std::log(node->visits + 1)
                            / (child->visits + 1e-9));
        double score = exploitation + C * exploration;
        if(score > bestScore){
            bestScore = score;
            best = child.get();
        }
    }
    return best;
}

void MCTS::expand(MCTSNode* node, const std::unordered_set<uint64_t>& history){
  if(node->expanded) return;
    node->expanded = true;

    bitboard legal = node->state.legalMoves(history);
    for(int i = 0; i < 6; ++i){
        uint64_t word = legal.w[i];
        while(word){
            int bit = __builtin_ctzll(word);
            int pos = i * 64 + bit;
            if(pos < 361){
                auto child = std::make_unique<MCTSNode>(
                    node->state.makeMove(pos), pos, node);
                node->children.push_back(std::move(child));
            }
            word &= word - 1;
        }
    }
    node->children.push_back(
        std::make_unique<MCTSNode>(node->state.makePass(), -1, node));
}

double MCTS::simulate(boardstate state){
      std::unordered_set<uint64_t> simHistory;
    int maxMoves = 500;
    static std::mt19937 rng(42);

    while(state.passCount < 2 && maxMoves-- > 0){
        // count empty squares
        int empties[361];
        int count = 0;
        for(int i = 0; i < 6; ++i){
            uint64_t bits = state.empty.w[i];
            while(bits){
                int bit = __builtin_ctzll(bits);
                int pos = i * 64 + bit;
                if(pos < 361) empties[count++] = pos;
                bits &= bits - 1;
            }
        }

        // shuffle indices and try legal moves
        for(int i = count - 1; i > 0; --i){
            int j = rng() % (i + 1);
            std::swap(empties[i], empties[j]);
        }

        bool moved = false;
        for(int k = 0; k < count; k++){
            int pos = empties[k];
            if(state.isLegal(pos, simHistory)){
                simHistory.insert(state.zobristHash);
                state = state.makeMove(pos);
                moved = true;
                break;
            }
        }
        if(!moved) state = state.makePass();
    }
    return state.score(komi) > 0 ? 1.0 : 0.0;
}

void MCTS::backpropagate(MCTSNode* node, double result){
    while(node != nullptr){
        node->visits++;
        node->wins += result;
        result = 1.0 - result; // flip perspective each level
        node = node->parent;
    }
}

