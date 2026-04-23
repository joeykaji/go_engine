#ifndef BOARDSTATE
#define BOARDSTATE

#include "bitboard.hpp"
#include <unordered_set>
#include "zobrist.hpp"
#include <cstdint>


struct boardstate{
  bitboard white;
  bitboard black;
  bitboard empty;
  bool blackMove;
  int passCount;
  uint64_t zobristHash;
  int blackCaptures;
  int whiteCaptures;
  uint16_t parent[361];
  uint8_t  liberties[361];
  uint16_t size[361];
  uint16_t next[361];
  
  boardstate();
  bitboard getGroup(int pos, const bitboard& color) const;
  bitboard getLiberties(bitboard group) const;
  void resolveCaptures(int pos);
  void recomputeGroupLiberties(int pos);
  bool isLegal(int pos, const std::unordered_set<uint64_t>& history) const;
  bool isLegalFast(int pos) const;
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
  bool isLegalNoKo(int pos) const {
    if (!(empty.w[pos / 64] & (1ULL << (pos % 64))))
        return false;
    const bitboard& friendly = blackMove ? black : white;
    const bitboard& enemy    = blackMove ? white : black;
    int nbrs[4];
    int nbCount = getNeighbors(pos, nbrs);
    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (empty.w[nb / 64] & (1ULL << (nb % 64)))         return true;
        if (friendly.w[nb / 64] & (1ULL << (nb % 64)))
            if (liberties[find(nb)] > 1)                     return true;
        if (enemy.w[nb / 64] & (1ULL << (nb % 64)))
            if (liberties[find(nb)] == 1)                    return true;
    }
    return false;
}
  

  bool isTerminal() const;
  float score(float komi) const;
};

#endif
