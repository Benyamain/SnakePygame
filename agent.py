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
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory of the agent (1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Train long memory then plot the result
            # Replay memory
            # Experienced replay
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                # agent.model.save()
            
            print('Game:', agent.n_games, 'Score:', score, 'Best Score:', best_score)

            # TODO: plot

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # To control the randomness of the game
        self.gamma = 0 # Discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # Popleft() if memory is exceeded
        self.model = None # TODO
        self.trainer = None # TODO

    def get_state(self, game):
        head = game.snake[0]
        # 20 is the BLOCK_SIZE
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        # Convert the booleans to an int
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # One tuple that we store due to the (())
        self.memory.append((state, action, reward, next_state, game_over)) # Popleft() if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Return a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        
        # Another way of doing it besides the zip function
        # for state, action, reward, next_state, game_over in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Do some random moves: Tradeoff between exploration / exploitation
        # Algorithm makes the snake explore freely until it reaches about 80 games through scope of randomness
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            # Set the [...,...,...]
            final_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model.predict(state_0)
            # Converts raw data to singular value
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

if __name__ == '__main__':
    train()