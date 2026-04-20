#ifndef BOARDSTATE
#define BOARDSTATE

#include "bitboard.hpp"
#include <unordered_set>
#include "zobrist.hpp"


struct boardstate{
  bitboard white;
  bitboard black;
  bitboard empty;
  bool blackMove;
  int passCount;
  uint64_t zobristHash;
  int blackCaptures;
  int whiteCaptures;
  int parent[361];      // union-find: group representative
  int liberties[361];   // liberty count per group root
  int size[361];        // group size
  int next[361];        // linked list of stones in group
  
  boardstate();
  bitboard getGroup(int pos, const bitboard& color) const;
  bitboard getLiberties(bitboard group) const;
  void resolveCaptures(int pos);
  void recomputeGroupLiberties(int pos);
  bool isLegal(int pos, const std::unordered_set<uint64_t>& history) const;
  bitboard legalMoves(const std::unordered_set<uint64_t>& history) const;

  boardstate makeMove(int pos) const;
  boardstate makePass() const;
  int getNeighbors(int pos, int* out) const {
    int count = 0;
    int row = pos / 19;
    int col = pos % 19;
    if (row > 0)  out[count++] = pos - 19;
    if (row < 18) out[count++] = pos + 19;
    if (col > 0)  out[count++] = pos - 1;
    if (col < 18) out[count++] = pos + 1;
    return count;
}
  int find(int pos) const {
    while (parent[pos] != pos)
        pos = parent[pos];
    return pos;
  }
  void mergeGroups(int a, int rootB) {
    int ra = find(a);
    if (ra == rootB) return;
    if (size[ra] < size[rootB]) std::swap(ra, rootB);
    parent[rootB] = ra;
    size[ra] += size[rootB];
    std::swap(next[ra], next[rootB]);
}
  

  bool isTerminal() const;
  float score(float komi) const;
};

#endif
