#ifndef BITBOARD
#define BITBOARD

#include <array>
#include <cstdint>

namespace detail{
  static constexpr std::array<uint64_t, 6> leftEdge(){
   int starter = 0;
    std::array<uint64_t, 6> returner = {};
    for(int i = 0; i < 6; ++i){
      for(int j = 0; j < 64; ++j){
        if(i * 64 + j >= 361){
          continue;
        }
        if(starter % 19 == 0){
          returner[i] |= (1ULL << j);
        }
        starter++;
      }
    }
    return returner;
  }

  static constexpr std::array<uint64_t, 6> rightEdge(){
    int starter = 0;
    std::array<uint64_t, 6> returner = {};
    for(int i = 0; i < 6; ++i){
      for(int j = 0; j < 64; ++j){
        if(i*64 + j >= 361){
          continue;
        }
        if(starter%19 == 18){
          returner[i] |= (1ULL << j);
        }
        starter += 1;
      }
    }
    return returner;
  }
}


struct bitboard{
  //Little Endian Style
  bitboard upShift(){
    bitboard ret;
    ret.w = w;
    uint64_t carry = 0;
    for(int i = 5; i >= 0; --i){
      uint64_t nextCarry = ret.w[i] << 45;
      ret.w[i] = ret.w[i] >> 19;
      ret.w[i] |= carry;
      carry = nextCarry;
    }
    return ret;
  }
  bitboard downShift(){
   //Each row is 19 so shift by 19 places
    bitboard ret;
    ret.w = w;
    uint64_t carry = 0;
    for(int i = 0; i < 6; ++i){
      uint64_t nextCarry = ret.w[i] >> 45;
      ret.w[i] = ret.w[i] << 19;
      ret.w[i] |= carry;
      carry = nextCarry;
    }
    return ret;
  }
  bitboard rightShift(){
    bitboard ret;
    ret.w = w;
    uint64_t carry = 0;
    for(int i = 5; i >= 0; --i){
      uint64_t nextCarry = ret.w[i] << 63;
      ret.w[i] = ret.w[i] >> 1;
      ret.w[i] |= carry;
      carry = nextCarry;
    }
    for(int i = 0; i < 6; ++i){
      ret.w[i] &= ~bitRightEdge[i];
    }
    return ret;
  }
  bitboard leftShift(){
    bitboard ret;
    ret.w = w;
    uint64_t carry = 0;
    for(int i = 0; i < 6; ++i){
      uint64_t nextCarry = ret.w[i] >> 63;
      ret.w[i] = ret.w[i] << 1;
      ret.w[i] |= carry; 
      carry = nextCarry;
    }
    for(int i = 0; i < 6; ++i){
      ret.w[i] &= ~bitLeftEdge[i];
    }
    return ret;
  }
  bitboard andBoard(std::array<uint64_t, 6> &m){
    bitboard ret;
    ret.w = w;
    for(int i = 0; i < 6; ++i){
      ret.w[i] &= m[i];
    }
    return ret;
  }
  bitboard orBoard(std::array<uint64_t, 6> &m){
    bitboard ret;
    ret.w = w;
    for(int i = 0; i < 6; ++i){
      ret.w[i] |= m[i];
    }
    return ret;
  }
  bitboard notBoard(){
    bitboard ret;
    ret.w = w;
    for(int i = 0; i < 6; ++i){
      ret.w[i] = ~ret.w[i];
    }
    return ret;
  }
  int countStones() const{
    int returner = 0;
    for(int i = 0; i < 6; ++i){
      returner += __builtin_popcountll(w[i]);
    }
    return returner;
  }
  bool getStone(int m, int n) const{
    int index =  m * 19 + n;
    uint64_t bit = (1ULL << index%64);
    int word = index/64;
    return bool(w[word] & bit);
  }
  void setStone(int m, int n){
    int index = m*19 + n;
    uint64_t bit = (1ULL << index%64);
    int word = index/64;
    w[word] |= bit;
  }
  void clearStone(int m, int n){
    int index = m* 19 + n;
    uint64_t bit = ~(1ULL << index%64);
    int word = index/64;
    w[word] &= bit;
  }
  std::array<uint64_t, 6> w;
  uint64_t validBits = (1ULL << 41) - 1; // Only for last
  static constexpr std::array<uint64_t, 6> bitLeftEdge = detail::leftEdge(); //Make sure you not it 
  static constexpr std::array<uint64_t, 6> bitRightEdge = detail::rightEdge(); //Make sure you not it
};

#endif
