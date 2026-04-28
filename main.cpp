#include "iostream"
#include "boardstate.hpp"
#include "zobrist.hpp"
#include <unordered_set>
#include "tests.cpp"
#include "mcts.hpp"

int main(){
  Zobrist::init();
  std::cout << sizeof(boardstate) << "\n";
  /*
  boardstate b;
  Zobrist::init();

  testCapture();
  testSuicide();
  testKo();
  testLegalMoves();

  int N = 1'000'000;
  auto start = std::chrono::high_resolution_clock::now();
  
  for(int i = 0; i < N; ++i){
    boardstate next = b.makeMove(20); // pick any valid pos
  }
    
  auto end = std::chrono::high_resolution_clock::now();
  double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "makeMove: " << ns / N << " ns/call\n";
  return 0;
  */
    boardstate b{};
    b.blackMove = true;
    std::unordered_set<uint64_t> history;
    MCTS mcts(1.41, 6.5f, "gonet.pt");
    while(b.passCount < 2){
      auto start = std::chrono::high_resolution_clock::now();
        int move = mcts.getBestMove(b, history, 200);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "200 iterations: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
          << "ms\n";
        history.insert(b.zobristHash);
        if(move == -1)
            b = b.makePass();
        else
            b = b.makeMove(move);
        std::cout << "Move: " << move << " Score: " << b.score(6.5f) << "\n";
    }
    std::cout << "Game over. Final score: " << b.score(6.5f) << "\n";
}
