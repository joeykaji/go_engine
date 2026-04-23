#include "bitboard.hpp"
#include <unordered_set>
#include "zobrist.hpp"
#include "boardstate.hpp"
#include <cassert>
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
    for (int i = 0; i < 361; i++) {
        parent[i] = i;
        liberties[i] = 0;
        size[i] = 1;
        next[i] = i;  // circular linked list, points to self
    }
}

void boardstate::recomputeGroupLiberties(int pos) {
    int root = find(pos);
    const bitboard& color = (black.w[pos/64] & (1ULL << (pos%64))) ? black : white;
    bitboard grp = getGroup(pos, color);
    liberties[root] = getLiberties(grp).countStones();
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
    bitboard& enemy    = blackMove ? white : black;
    bitboard& friendly = blackMove ? black : white;
    int& captureCount  = blackMove ? blackCaptures : whiteCaptures;
 
    int nbrs[4];
    int nbCount = getNeighbors(pos, nbrs);
 
    // collect roots of friendly groups adjacent to pos — we'll recompute their
    // liberties after all captures are resolved
    int friendlyRoots[4];
    int friendlyRootCount = 0;
 
    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (!(enemy.w[nb / 64] & (1ULL << (nb % 64)))) {
            // track friendly neighbors for later recompute
            if (friendly.w[nb / 64] & (1ULL << (nb % 64))) {
                int r = find(nb);
                bool seen = false;
                for(int k = 0; k < friendlyRootCount; k++)
                    if(friendlyRoots[k] == r) { seen = true; break; }
                if(!seen) friendlyRoots[friendlyRootCount++] = r;
            }
            continue;
        }
 
        int root = find(nb);
        // use liberties array to check — but now it's always correct since we
        // recompute after every move, so this check is valid
        if (liberties[root] != 0) continue;
 
        // collect dying stones first, then process
        int dying_stones[361];
        int dying_count = 0;
        int cur = root;
        do {
            dying_stones[dying_count++] = cur;
            cur = next[cur];
        } while (cur != root);
        captureCount += dying_count;
 
        for (int d = 0; d < dying_count; d++) {
            int dying = dying_stones[d];
 
            zobristHash ^= blackMove ? Zobrist::white[dying] : Zobrist::black[dying];
            enemy.w[dying / 64] &= ~(1ULL << (dying % 64));
 
            // reset union-find for this stone
            parent[dying]    = dying;
            size[dying]      = 1;
            next[dying]      = dying;
            liberties[dying] = 0;
        }
    }
 
    // keep empty in sync
    for (int i = 0; i < 6; ++i)
        empty.w[i] = ~(black.w[i] | white.w[i]);
    empty.w[5] &= (1ULL << 41) - 1;
 
    // recompute liberties for friendly groups that were adjacent to captures
    for(int i = 0; i < friendlyRootCount; i++)
        recomputeGroupLiberties(friendlyRoots[i]);
}

bool boardstate::isLegal(int pos, const std::unordered_set<uint64_t>& history) const {
    if (!(empty.w[pos / 64] & (1ULL << (pos % 64))))
        return false;

    const bitboard& friendly = blackMove ? black : white;
    const bitboard& enemy    = blackMove ? white : black;

    int nbrs[4];
    int nbCount = getNeighbors(pos, nbrs);

    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (empty.w[nb / 64] & (1ULL << (nb % 64)))
            goto kocheck;                          // direct liberty
        if (friendly.w[nb / 64] & (1ULL << (nb % 64)))
            if (liberties[find(nb)] > 1)
                goto kocheck;                      // connects to safe group
        if (enemy.w[nb / 64] & (1ULL << (nb % 64)))
            if (liberties[find(nb)] == 1)
                goto kocheck;                      // captures enemy
    }
    return false;                                  // suicide

kocheck:
    boardstate result = makeMove(pos);
    return !history.count(result.zobristHash);
}

bool boardstate::isLegalFast(int pos) const {
    if (!(empty.w[pos/64] & (1ULL << (pos%64)))) return false;
    // skip ko check — acceptable for rollouts
    const bitboard& friendly = blackMove ? black : white;
    const bitboard& enemy    = blackMove ? white : black;
    int nbrs[4]; int nbCount = getNeighbors(pos, nbrs);
    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (empty.w[nb/64] & (1ULL << (nb%64))) return true;
        if (friendly.w[nb/64] & (1ULL << (nb%64)))
            if (liberties[find(nb)] > 1) return true;
        if (enemy.w[nb/64] & (1ULL << (nb%64)))
            if (liberties[find(nb)] == 1) return true;
    }
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

boardstate boardstate::makeMove(int pos) const {
    boardstate next = *this;
 
    bitboard& friendly = next.blackMove ? next.black : next.white;
 
    // place stone
    friendly.w[pos / 64] |= (1ULL << (pos % 64));
    next.empty.w[pos / 64] &= ~(1ULL << (pos % 64));
    next.zobristHash ^= next.blackMove ? Zobrist::black[pos] : Zobrist::white[pos];
 
    // init new stone as its own group
    next.parent[pos]    = pos;
    next.size[pos]      = 1;
    next.next[pos]      = pos;
    next.liberties[pos] = 0;
 
    // merge with adjacent friendly groups
    int nbrs[4];
    int nbCount = getNeighbors(pos, nbrs);
 
    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (friendly.w[nb / 64] & (1ULL << (nb % 64))) {
            next.mergeGroups(pos, next.find(nb));
        }
    }
    // resolve captures (removes dead enemy groups, resyncs empty,
    // recomputes liberties for friendly groups adjacent to captures)
    next.resolveCaptures(pos);
 
    // recompute liberties for the placed stone's merged group
    // (resolveCaptures handles groups affected by captures, but
    //  the placed group itself needs recompute for its own liberties)
    next.recomputeGroupLiberties(pos);
 
    // also recompute any surviving enemy groups adjacent to pos
    // (their liberties decreased by 1 due to the placed stone)
    const bitboard& enemy = next.blackMove ? next.black : next.white;
    for (int i = 0; i < nbCount; i++) {
        int nb = nbrs[i];
        if (enemy.w[nb / 64] & (1ULL << (nb % 64)))
            next.recomputeGroupLiberties(nb);
    }
 
    next.zobristHash ^= Zobrist::sideToMove;
    next.blackMove = !next.blackMove;
    next.passCount = 0;
 
#ifndef NDEBUG
    const bitboard& f = next.blackMove ? next.white : next.black;
    bitboard grp = next.getGroup(pos, f);
    bitboard libs = next.getLiberties(grp);
    if(libs.countStones() != next.liberties[next.find(pos)]){
        std::cout << "pos: " << pos << " (" << pos/19 << "," << pos%19 << ")\n";
        std::cout << "expected liberties: " << libs.countStones() << "\n";
        std::cout << "got liberties:      " << next.liberties[next.find(pos)] << "\n";
        std::cout << "diff: " << (int)next.liberties[next.find(pos)] - (int)libs.countStones() << "\n";
    }
    assert(libs.countStones() == next.liberties[next.find(pos)]);
#endif
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
    float blackScore = black.countStones();
    float whiteScore = white.countStones();

    // count empty points surrounded by only one color
    for (int i = 0; i < 361; i++) {
        if (!(empty.w[i/64] & (1ULL << (i%64)))) continue;
        // flood fill from this empty point
        bool touchesBlack = false, touchesWhite = false;
        // check 4 neighbors
        int nbrs[4];
        int nbCount = getNeighbors(i, nbrs);
        for (int k = 0; k < nbCount; k++) {
            int nb = nbrs[k];
            if (black.w[nb/64] & (1ULL << (nb%64))) touchesBlack = true;
            if (white.w[nb/64] & (1ULL << (nb%64))) touchesWhite = true;
        }
        if (touchesBlack && !touchesWhite) blackScore++;
        else if (touchesWhite && !touchesBlack) whiteScore++;
    }

    return blackScore - whiteScore - komi;
  }


