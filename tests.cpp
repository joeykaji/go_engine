#include "iostream"
#include "boardstate.hpp"
#include "zobrist.hpp"
#include <unordered_set>

void printBoard(const boardstate& b){
    std::cout << "   ";
    for(int col = 0; col < 19; ++col){
        std::cout << (char)('A' + col) << " ";
    }
    std::cout << "\n";

    for(int row = 0; row < 19; ++row){
        std::cout << (row + 1 < 10 ? " " : "") << row + 1 << " ";
        for(int col = 0; col < 19; ++col){
            if(b.black.getStone(row, col))       std::cout << "X ";
            else if(b.white.getStone(row, col))  std::cout << "O ";
            else                                  std::cout << ". ";
        }
        std::cout << "\n";
    }
    std::cout << "Black captures: " << b.blackCaptures << "\n";
    std::cout << "White captures: " << b.whiteCaptures << "\n";
    std::cout << "Hash: " << b.zobristHash << "\n";
    std::cout << "\n" << std::endl;
}

void testCapture(){
    std::cout << "=== TEST: Basic capture ===\n";
    // surround a white stone at (0,1) and capture it
    // black plays (0,0), (0,2), (1,1) — white plays (0,1)

    boardstate b{};
    std::unordered_set<uint64_t> history;

    b.blackMove = true;
    b = b.makeMove(0 * 19 + 0);  // black (0,0)
    history.insert(b.zobristHash);

    b = b.makeMove(0 * 19 + 1);  // white (0,1) ERROR IS HERE
    history.insert(b.zobristHash);

    b = b.makeMove(0 * 19 + 2);  // black (0,2)
    history.insert(b.zobristHash);

    b = b.makeMove(1 * 19 + 0);  // white somewhere safe
    history.insert(b.zobristHash);

    b = b.makeMove(1 * 19 + 1);  // black (1,1) — captures white at (0,1)
    history.insert(b.zobristHash);

    printBoard(b);
    std::cout << "Expected: white stone at (0,1) gone, blackCaptures = 1\n\n" << std::endl;
}

void testSuicide(){
    std::cout << "=== TEST: Suicide is illegal ===\n";
    // black surrounds (0,0) so white can't play there
    boardstate b{};
    std::unordered_set<uint64_t> history;

    b.blackMove = true;
    b = b.makeMove(0 * 19 + 1);  // black (0,1)
    history.insert(b.zobristHash);

    b = b.makeMove(18 * 19 + 0); // white somewhere safe
    history.insert(b.zobristHash);

    b = b.makeMove(1 * 19 + 0);  // black (1,0)
    history.insert(b.zobristHash);

    // now white playing (0,0) would be suicide
    bool legal = b.isLegal(0 * 19 + 0, history);
    std::cout << "White playing (0,0) legal: " << (legal ? "YES (wrong!)" : "NO (correct)") << "\n\n";
}

void testKo(){
    std::cout << "=== TEST: Ko ===\n";
    // classic ko position
    // . X O .
    // X . X O
    // . X O .
    boardstate b{};
    std::unordered_set<uint64_t> history;

    b.blackMove = true;

    // set up ko position manually
    b.black.setStone(0, 1);
    b.black.setStone(1, 0);
    b.black.setStone(1, 2);
    b.black.setStone(2, 1);

    b.white.setStone(0, 2);
    b.white.setStone(1, 3);
    b.white.setStone(2, 2);
    // white at (1,1) is already captured, black to retake at (1,1)

    // sync empty
    for(int i = 0; i < 6; ++i){
        b.empty.w[i] = ~(b.black.w[i] | b.white.w[i]);
    }
    b.empty.w[5] &= (1ULL << 41) - 1;

    printBoard(b);

    // black captures at (1,1)
    history.insert(b.zobristHash);
    b = b.makeMove(1 * 19 + 1);
    history.insert(b.zobristHash);
    printBoard(b);

    // white trying to retake at (1,2) — should be illegal due to ko
    bool legal = b.isLegal(1 * 19 + 2, history);
    std::cout << "White retaking ko legal: " << (legal ? "YES (wrong!)" : "NO (correct)") << "\n\n";
}

void testLegalMoves(){
    std::cout << "=== TEST: Legal moves on empty board ===\n";
    boardstate b{};
    for(int i = 0; i < 6; ++i){
        b.empty.w[i] = ~0ULL;
    }
    b.empty.w[5] &= (1ULL << 41) - 1;
    b.blackMove = true;

    std::unordered_set<uint64_t> history;
    bitboard moves = b.legalMoves(history);
    int count = moves.countStones();
    std::cout << "Legal moves on empty board: " << count << " (expected 361)\n\n";
}

