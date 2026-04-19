#ifndef ZOBRIST
#define ZOBRIST

#include <cstdint>
#include <array>
#include <random>


namespace Zobrist {
    // one value per position per color, plus side to move
    inline std::array<uint64_t, 361> black;
    inline std::array<uint64_t, 361> white;
    inline uint64_t sideToMove;

    inline void init() {
        std::mt19937_64 rng(12345678ULL);  // fixed seed for reproducibility
        for(int i = 0; i < 361; ++i){
            black[i] = rng();
            white[i] = rng();
        }
        sideToMove = rng();
    }
}

#endif
