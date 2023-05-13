import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Get the old state or current state
        state_old = agent.get_state(game)

        # Get the move based on current state
        final_move = agent.get_action(state_old)

        # Perform the move and get a new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory of the agent (1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory then plot the result
            # Replay memory
            # Experienced replay
            game.reset()


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # To control the randomness of the game
        self.gamma = 0 # Discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # Popleft() if memory is exceeded
        # TODO: model, trainer

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, game_over):
        pass

    def get_action(self, state):
        pass

if __name__ == '__main__':
    train()