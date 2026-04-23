import torch
import torch.nn as nn
import numpy as np
import random
import sys
sys.path.insert(0, '.')
from network import GoNet

def random_game(max_moves=200):
    """Play a random game and return (states, winner)"""
    # We'll represent board as two 19x19 numpy arrays
    black = np.zeros((19,19), dtype=np.float32)
    white = np.zeros((19,19), dtype=np.float32)
    black_move = True
    pass_count = 0
    states = []
    moves = []
    
    for _ in range(max_moves):
        # record state
        plane = np.stack([black, white, 
                         np.full((19,19), 1.0 if black_move else 0.0)])
        states.append(plane)
        
        # pick random empty cell or pass
        empty = [(r,c) for r in range(19) for c in range(19) 
                 if black[r,c] == 0 and white[r,c] == 0]
        
        if not empty or random.random() < 0.05:
            moves.append(361)  # pass
            pass_count += 1
            if pass_count >= 2:
                break
        else:
            r, c = random.choice(empty)
            moves.append(r * 19 + c)
            if black_move:
                black[r,c] = 1
            else:
                white[r,c] = 1
            pass_count = 0
        
        black_move = not black_move
    
    # simple score: count stones
    winner = 1.0 if black.sum() > white.sum() + 6.5 else 0.0
    return states, moves, winner

def generate_dataset(num_games=1000):
    all_states, all_policies, all_values = [], [], []
    
    for i in range(num_games):
        if i % 100 == 0:
            print(f"Game {i}/{num_games}")
        states, moves, winner = random_game()
        
        for j, (state, move) in enumerate(zip(states, moves)):
            policy = np.zeros(362, dtype=np.float32)
            policy[move] = 1.0
            # value from perspective of current player
            value = winner if j % 2 == 0 else 1.0 - winner
            
            all_states.append(state)
            all_policies.append(policy)
            all_values.append(value)
    
    return (np.array(all_states), 
            np.array(all_policies), 
            np.array(all_values))

if __name__ == "__main__":
    print("Generating games...")
    states, policies, values = generate_dataset(1000)
    np.save("states.npy", states)
    np.save("policies.npy", policies)
    np.save("values.npy", values)
    print(f"Saved {len(states)} positions")
