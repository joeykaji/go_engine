#include "bitboard.hpp"
#include <unordered_set>
#include "zobrist.hpp"
#include "boardstate.hpp"
#include <iostream>

boardstate::boardstate() {
    black  = {};
    white  = {};
    empty.w[0] = ~0ULL;           // bits 0-63
    empty.w[1] = ~0ULL;           // bits 64-127
    empty.w[2] = ~0ULL;           // bits 128-191
    empty.w[3] = ~0ULL;           // bits 192-255
    empty.w[4] = ~0ULL;           // bits 256-319
    empty.w[5] = (1ULL << 41) - 1; // bits 320-360, only 41 bits valid
    blackMove = true;
    passCount = 0;
    blackCaptures = 0;
    whiteCaptures = 0;
    zobristHash = 0;
}

bitboard boardstate::getGroup(int pos, const bitboard& color) const{
    bitboard group{};
    int word = pos / 64;
    int bit  = pos % 64;
    group.w[word] = (1ULL << bit);

    while(true){
      bitboard expanded = group;
      
      bitboard u = group.upShift();
      bitboard d = group.downShift();
      bitboard l = group.leftShift();
      bitboard r = group.rightShift();
      for(int i = 0; i < 6; ++i){
        expanded.w[i] |= u.w[i] | d.w[i] | l.w[i] | r.w[i];
        expanded.w[i] &= color.w[i];  // only keep stones of same color
      }
      
      bool changed = false;
      for(int i = 0; i < 6; ++i){
        if(expanded.w[i] != group.w[i]){
          changed = true;
          break;
        }
      }
      if(!changed) break;
      group = expanded;

    }
    return group;
  }

bitboard boardstate::getLiberties(bitboard group) const {
    bitboard neighbors{};
    for(int i = 0; i < 6; ++i){
        neighbors.w[i] = group.upShift().w[i]
                       | group.downShift().w[i]
                       | group.leftShift().w[i]
                       | group.rightShift().w[i];
    }

    bitboard liberties{};
    for(int i = 0; i < 6; ++i)
        liberties.w[i] = neighbors.w[i] & empty.w[i];

    return liberties;
}

void boardstate::resolveCaptures(int pos) {
    bitboard& enemy = blackMove ? white : black;
    int& captureCount = blackMove ? blackCaptures : whiteCaptures;

    bitboard placed{};
    placed.w[pos / 64] = (1ULL << (pos % 64));

    bitboard neighbors{};
    for(int i = 0; i < 6; ++i){
        neighbors.w[i] = placed.upShift().w[i]
                       | placed.downShift().w[i]
                       | placed.leftShift().w[i]
                       | placed.rightShift().w[i];
    }

    bitboard toCheck{};
    for(int i = 0; i < 6; ++i)
        toCheck.w[i] = neighbors.w[i] & enemy.w[i];

    while(true){
        int foundPos = -1;
        for(int i = 0; i < 6; ++i){
            if(toCheck.w[i]){
                foundPos = i * 64 + __builtin_ctzll(toCheck.w[i]);
                break;
            }
        }
        if(foundPos == -1) break;

        bitboard group = getGroup(foundPos, enemy);
        bitboard liberties = getLiberties(group);

        bool dead = true;
        for(int i = 0; i < 6; ++i){
            if(liberties.w[i]){ dead = false; break; }
        }

        if(dead){
          captureCount += group.countStones();
            for(int i = 0; i < 6; ++i){
                uint64_t dying = group.w[i];
                while(dying){
                    int bit = __builtin_ctzll(dying);
                    int capturedPos = i * 64 + bit;
                    if(blackMove){
                        zobristHash ^= Zobrist::white[capturedPos];
                    } else {
                        zobristHash ^= Zobrist::black[capturedPos];
                    }
                    dying &= dying - 1;
                }
                enemy.w[i] &= ~group.w[i];
            }
        }

        for(int i = 0; i < 6; ++i)
            toCheck.w[i] &= ~group.w[i];
    }

    // keep empty in sync
    for(int i = 0; i < 6; ++i)
        empty.w[i] = ~(black.w[i] | white.w[i]);
    // mask off invalid bits in last word
    empty.w[5] &= (1ULL << 41) - 1;
  }

bool boardstate::isLegal(int pos, const std::unordered_set<uint64_t>& history) const {
    if(!(empty.w[pos / 64] & (1ULL << (pos % 64))))
        return false;
    boardstate next = makeMove(pos);
    const bitboard& friendly = next.blackMove ? next.white : next.black;
    bitboard group = next.getGroup(pos, friendly);
    bitboard liberties = next.getLiberties(group);
    for(int i = 0; i < 6; ++i)
        if(liberties.w[i])
            return !history.count(next.zobristHash);
    return false;
}

bitboard boardstate::legalMoves(const std::unordered_set<uint64_t>& history) const {
    bitboard moves;
    for(int i = 0; i < 6; ++i)
        moves.w[i] = 0;

    for(int i = 0; i < 6; ++i){
        uint64_t word = empty.w[i];
        while(word){
            int bit = __builtin_ctzll(word);
            int pos = i * 64 + bit;
            if(pos < 361 && isLegal(pos, history))
                moves.w[i] |= (1ULL << bit);
            word &= word - 1;  // clear lowest set bit
        }
    }
    return moves;
  }
bool boardstate::isLegalFast(int pos) const {
    if(!(empty.w[pos / 64] & (1ULL << (pos % 64))))
        return false;
    // skip superko for MCTS expansion, just check suicide
    boardstate next = *this;
    bitboard& friendly = next.blackMove ? next.black : next.white;
    friendly.w[pos / 64] |= (1ULL << (pos % 64));
    next.empty.w[pos / 64] &= ~(1ULL << (pos % 64));
    next.resolveCaptures(pos);
    const bitboard& f = next.blackMove ? next.black : next.white;
    bitboard group = next.getGroup(pos, f);
    bitboard liberties = next.getLiberties(group);
    for(int i = 0; i < 6; ++i)
        if(liberties.w[i]) return true;
    return false;
}

boardstate boardstate::makeMove(int pos) const {
    boardstate next = *this;

    bitboard& friendly = next.blackMove ? next.black : next.white;
    friendly.w[pos / 64] |= (1ULL << (pos % 64));
    next.empty.w[pos / 64] &= ~(1ULL << (pos % 64));

    if(next.blackMove){
      next.zobristHash ^= Zobrist::black[pos];
    } else {
    next.zobristHash ^= Zobrist::white[pos];
    }
    
    next.resolveCaptures(pos);

    // flip side to move at end of makeMove and makePass:
    next.zobristHash ^= Zobrist::sideToMove;

    next.blackMove = !next.blackMove;
    next.passCount = 0;

    return next;
  }

boardstate boardstate::makePass() const {
    boardstate next = *this;
    next.zobristHash ^= Zobrist::sideToMove;
    next.blackMove = !next.blackMove;
    next.passCount++;
    return next;
  }

bool boardstate::isTerminal() const {
    return passCount >= 2;
  }

float boardstate::score(float komi) const {
    // simple territory count using flood fill from empty points
    // for now just count stones (area scoring)
    float blackScore = black.countStones();
    float whiteScore = white.countStones() + komi;
    return blackScore - whiteScore;  // positive = black wins
  }


