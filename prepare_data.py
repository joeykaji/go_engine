"""
prepare_data.py — memory-safe version using np.memmap
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
import json

# ── SGF parser ────────────────────────────────────────────────────────────────

def parse_sgf(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sz = re.search(r"SZ\[(\d+)\]", text)
    if not sz or int(sz.group(1)) != 19:
        return None, None
    re_match = re.search(r"RE\[([^\]]+)\]", text)
    winner = None
    if re_match:
        result = re_match.group(1)
        if result.startswith("B"):   winner = "B"
        elif result.startswith("W"): winner = "W"
    if winner is None:
        return None, None
    moves = []
    for m in re.finditer(r";([BW])\[([a-z]{0,2})\]", text):
        color = m.group(1)
        coord = m.group(2)
        if coord == "" or coord == "tt":
            moves.append((color, None))
        else:
            col = ord(coord[0]) - ord("a")
            row = ord(coord[1]) - ord("a")
            if 0 <= row < 19 and 0 <= col < 19:
                moves.append((color, (row, col)))
    return moves, winner


# ── board ─────────────────────────────────────────────────────────────────────

class SimpleBoard:
    def __init__(self):
        self.board = np.zeros((19, 19), dtype=np.int8)
        self.turn  = 1

    def encode(self, out, winner):
        if self.turn == 1:  # black to play
            out[0] = (self.board == 1)  # current player
            out[1] = (self.board == 2)  # opponent
            out[2] = 1.0
            return 1.0 if winner == "B" else -1.0
        else:  # white to play
            out[0] = (self.board == 2)  # current player
            out[1] = (self.board == 1)  # opponent
            out[2] = 0.0
            return 1.0 if winner == "W" else -1.0 

    def get_group_and_liberties(self, row, col):
        color = self.board[row, col]
        if color == 0:
            return set(), set()
        visited, liberties = set(), set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited: continue
            visited.add((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < 19 and 0 <= nc < 19:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif self.board[nr, nc] == color and (nr,nc) not in visited:
                        stack.append((nr, nc))
        return visited, liberties

    def play(self, color_int, pos):
        if pos is None:
            self.turn = 3 - color_int
            return True
        row, col = pos
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = color_int
        enemy = 3 - color_int
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = row+dr, col+dc
            if 0 <= nr < 19 and 0 <= nc < 19 and self.board[nr,nc] == enemy:
                grp, libs = self.get_group_and_liberties(nr, nc)
                if len(libs) == 0:
                    for r, c in grp:
                        self.board[r, c] = 0
        _, libs = self.get_group_and_liberties(row, col)
        if len(libs) == 0:
            self.board[row, col] = 0
            self.turn = 3 - color_int
            return False
        self.turn = 3 - color_int
        return True


def pos_to_idx(pos):
    return 361 if pos is None else pos[0] * 19 + pos[1]


# ── pass 1: count positions ───────────────────────────────────────────────────

def count_positions(sgf_files):
    total = 0
    valid = []
    for path in sgf_files:
        moves, winner = parse_sgf(path)
        if moves is None or len(moves) < 10:
            continue
        total += len(moves)
        valid.append(path)
    return total, valid


# ── pass 2: encode and write ──────────────────────────────────────────────────

def encode_all(valid_files, total_positions, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Allocating memmaps for {total_positions} positions...")

    states   = np.memmap(out_dir / "states.npy",   dtype="float32", mode="w+",
                         shape=(total_positions, 3, 19, 19))
    policies = np.memmap(out_dir / "policies.npy", dtype="int64",   mode="w+",
                         shape=(total_positions,))
    values   = np.memmap(out_dir / "values.npy",   dtype="float32", mode="w+",
                         shape=(total_positions,))

    idx = 0
    color_map = {"B": 1, "W": 2}

    for game_num, path in enumerate(valid_files):
        moves, winner = parse_sgf(path)
        value = 1.0 if winner == "B" else -1.0

        board = SimpleBoard()
        for color_str, pos in moves:
            values[idx] = board.encode(states[idx], winner)  # encode returns value
            policies[idx] = pos_to_idx(pos)
            idx += 1
            board.play(color_map[color_str], pos) 

        if game_num % 1000 == 0:
            print(f"  {game_num}/{len(valid_files)} games  "
                  f"({idx} positions)...")
            # flush periodically to avoid OS buffer buildup
            states.flush()
            policies.flush()
            values.flush()

    # final flush
    states.flush()
    policies.flush()
    values.flush()

    print(f"\nDone. {idx} positions written.")
    print(f"states:   {states.shape}  "
          f"{states.nbytes / 1e9:.1f} GB")
    print(f"policies: {policies.shape}")
    print(f"values:   {values.shape}")
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"total_positions": idx}, f)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out_dir",   default=".")
    parser.add_argument("--max_games", type=int, default=None)
    args = parser.parse_args()

    sgf_files = list(Path(args.data_dir).rglob("*.sgf"))
    if args.max_games:
        sgf_files = sgf_files[:args.max_games]
    print(f"Found {len(sgf_files)} SGF files")

    print("Pass 1: counting positions...")
    total_positions, valid_files = count_positions(sgf_files)
    print(f"  {len(valid_files)} valid games, "
          f"{total_positions} total positions")

    print("Pass 2: encoding...")
    encode_all(valid_files, total_positions, args.out_dir)
