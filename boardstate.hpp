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
  
  boardstate();
  bitboard getGroup(int pos, const bitboard& color) const;
  bitboard getLiberties(bitboard group) const;
  void resolveCaptures(int pos);
  bool isLegal(int pos, const std::unordered_set<uint64_t>& history) const;
  bool isLegalFast(int pos) const;
  bitboard legalMoves(const std::unordered_set<uint64_t>& history) const;

  boardstate makeMove(int pos) const;
  boardstate makePass() const;

  bool isTerminal() const;
  float score(float komi) const;
};

#endif
