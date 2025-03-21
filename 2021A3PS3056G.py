import sys
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from TicTacToe import *
import matplotlib.pyplot as plt
from time import time
import os

class PlayerSQN:
    def __init__(self, mode='eval'):
        """
        Initializes the PlayerSQN class.
        """
        self.state_size = 9
        self.action_size = 9
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.batch_size = 64
        self.mode = mode
        self.model_path = '2021A3PS3056G_MODEL.h5'
        
        if mode == 'train':
            self.is_training = True
            print("Initializing new model for training.")
            self.model = self.set_QNN()
            self.training_history = {
                'losses': [],           
                'win_rates': [],        
                'q_values': [],         
                'q_targets': [],        
                'q_predictions': [],    
                'rewards': [],          
                'epsilon_values': [],   
                'episode_lengths': [],  
            }
        else:
            self.is_training = False
            if os.path.exists(self.model_path):
                print("Loading pre-trained model for evaluation.")
                try:
                    self.model = self.set_QNN()
                    self.model = load_model(self.model_path)
                    print("Model loaded successfully.")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Creating new model instead.")
                    self.model = self.set_QNN()
            else:
                raise FileNotFoundError("No pre-trained model found.")
    
    def set_QNN(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=MeanSquaredError(), 
                     optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def change_state_val(self, state):
        state_array = np.array(state, dtype=np.float32)
        state_array = np.where(state_array == 1, -1, state_array)
        state_array = np.where(state_array == 2, 1, state_array)
        return state_array
    
    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.

        Parameters:
        state (list): A list representing the current state of the TicTacToe board.

        Returns:
        int: The position (0-8) where Player 2 wants to make a move.
        """
        empty_cells = [i for i in range(9) if state[i] == 0]
        
        if self.is_training and random.random() < self.epsilon:
            return random.choice(empty_cells)
            
        state_array = self.change_state_val(state)
        q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)[0]
        
        valid_q_values = {move: q_values[move] for move in empty_cells}
        return max(valid_q_values.items(), key=lambda x: x[1])[0]
    
    def remember(self, state, action, reward, next_state, done):
        if self.is_training:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if not self.is_training or len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        # print(f"minibatch: {minibatch}")
        batch_loss = 0
        states = []
        targets = []
        
        batch_q_targets = []
        batch_q_predictions = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                state_array = self.change_state_val(next_state)
                next_q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)[0]
                # print(f"next_q_values: {next_q_values}")
                empty_cells = [i for i in range(9) if next_state[i] == 0]
                if empty_cells:
                    valid_q_values = [next_q_values[i] for i in empty_cells]
                    target = reward + self.gamma * max(valid_q_values)
                    # print(f"valid_q_values: {valid_q_values}")
            
            state_array = self.change_state_val(state)
            predicted_val = self.model.predict(state_array.reshape(1, -1), verbose=0)
            
            batch_q_targets.append(target)
            batch_q_predictions.append(predicted_val[0][action])
            predicted_val[0][action] = target
            predicted_val = predicted_val[0]
            
            # Give -1 to all the used positions
            for i in range(9):
                if (state_array[i] != 0): predicted_val[i] = -1
                pass

            game = TicTacToe()
            game.board = state

            # Give 0.9 to block opponent
            for position in game.empty_positions():
                game.board[position] = 2
                if game.check_winner(2):
                    predicted_val[position] = 0.9
                game.board[position] = 0

            # Give +1 if we are winning
            for position in game.empty_positions():
                game.board[position] = 1
                if game.check_winner(1):
                    predicted_val[position] = 1
                game.board[position] = 0

            # print(game.print_board())
            # print(f"predicted_val: {predicted_val}")
            states.append(state_array)
            targets.append(predicted_val)
        
        # print(f"states: {states}")
        # print(f"targets: {targets}")

        history = self.model.fit(np.array(states), np.array(targets), 
                               batch_size=self.batch_size, epochs=1, verbose=0)
        # print(f"history: {history.history}")
        batch_loss = history.history['loss'][0]
        
        if self.mode == 'train':
            self.training_history['losses'].append(batch_loss)
            self.training_history['q_targets'].append(np.mean(batch_q_targets))
            self.training_history['q_predictions'].append(np.mean(batch_q_predictions))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return batch_loss
    
    def plot_training_metrics(self):
        if self.mode != 'train':
            return
            
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 4, 1)
        plt.plot(self.training_history['losses'])
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        plt.subplot(2, 4, 2)
        plt.plot(self.training_history['win_rates'])
        plt.title('Win Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        
        plt.subplot(2, 4, 3)
        plt.plot(self.training_history['q_values'])
        plt.title('Average Q-Values')
        plt.xlabel('Episodes')
        plt.ylabel('Q-Value')
        
        plt.subplot(2, 4, 4)
        plt.plot(self.training_history['rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        
        plt.subplot(2, 4, 5)
        plt.plot(self.training_history['epsilon_values'])
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        
        plt.subplot(2, 4, 6)
        plt.plot(self.training_history['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episodes')
        plt.ylabel('Number of Moves')
        
        plt.subplot(2, 4, 7)
        if len(self.training_history['q_targets']) > 0:
            plt.plot(self.training_history['q_targets'], label='Q-Target')
            plt.plot(self.training_history['q_predictions'], label='Q-Prediction')
            plt.title('Q-Target vs Q-Prediction')
            plt.xlabel('Training Steps')
            plt.ylabel('Q-Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('2021A3PS3056G_training_metrics.png')
        plt.close()
    
    def save_model(self):
        if self.mode == 'train':
            self.model.save(self.model_path)
            print(f"Model saved as {self.model_path}")

def train_agent(smartMovePlayer1):
    agent = PlayerSQN(mode='train')
    print("Starting training")
    n_episodes = 2000
    collection_episodes = 500 
    training_frequency = 5
    
    print("Phase 1: Collecting initial experiences")
    for episode in range(collection_episodes):
        game = TicTacToe(smartMovePlayer1=0, playerSQN=agent)
        moves = 0
        done = False
        
        while not done:
            if not game.is_full() and game.current_winner is None:
                game.player1_move()
                moves += 1
                if game.current_winner == 1 or game.is_full():
                    done = True
            
            if not done:
                current_state = game.board.copy()
                action = agent.move(current_state)
                moves += 1
                
                game.make_move(action, 2)
                next_state = game.board.copy()
                reward = game.get_reward()
                done = game.current_winner is not None or game.is_full()
                
                if not done:
                    reward = -0.01
                elif game.current_winner == 2:
                    reward = 1.0
                elif game.current_winner == 1:
                    reward = -1.0
                else:
                    reward = 0.0
                
                agent.remember(current_state, action, reward, next_state, done)
        
        if (episode + 1) % 10 == 0:
            print(f"Collected data from episode {episode + 1}/{collection_episodes}")
    
    print("\nPhase 2: Training on collected experiences")
    current_difficulty = 0
    window_size = 100
    consecutive_wins = 0
    
    for episode in range(5000):
        if episode % training_frequency == 0:
            episode_loss = 0
            if len(agent.memory) >= agent.batch_size:
                for _ in range(10):
                    episode_loss += agent.replay()
                episode_loss /= 10
        
        game = TicTacToe(smartMovePlayer1=current_difficulty, playerSQN=agent)
        moves = 0
        done = False
        episode_q_values = []
        
        while not done:
            if not game.is_full() and game.current_winner is None:
                game.player1_move()
                moves += 1
                if game.current_winner == 1 or game.is_full():
                    done = True
            
            if not done:
                current_state = game.board.copy()
                state_array = agent.change_state_val(current_state)
                q_values = agent.model.predict(state_array.reshape(1, -1), verbose=0)[0]
                episode_q_values.append(np.mean(q_values))
                
                action = agent.move(current_state)
                moves += 1
                
                game.make_move(action, 2)
                next_state = game.board.copy()
                reward = game.get_reward()
                done = game.current_winner is not None or game.is_full()
                
                if not done:
                    reward = -0.01
                elif game.current_winner == 2:
                    reward = 1.0
                elif game.current_winner == 1:
                    reward = -1.0
                else:
                    reward = 0.0
                
                agent.remember(current_state, action, reward, next_state, done)
        
        agent.training_history['episode_lengths'].append(moves)
        agent.training_history['rewards'].append(reward)
        agent.training_history['epsilon_values'].append(agent.epsilon)
        if episode_q_values:
            agent.training_history['q_values'].append(np.mean(episode_q_values))
        
        if reward == 1:
            consecutive_wins += 1
        else:
            consecutive_wins = 0
        
        recent_rewards = agent.training_history['rewards'][-window_size:]
        win_rate = sum(1 for r in recent_rewards if r == 1) / len(recent_rewards)
        agent.training_history['win_rates'].append(win_rate)
        
        if (episode + 1) % 100 == 0:
            elapsed_time = time() - start_time
            print(f"\nEpisode {episode + 1}/{n_episodes} ({elapsed_time:.1f}s)")
            print(f"Win Rate: {win_rate:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Average Loss: {episode_loss:.6f}")
            print(f"Average Q-Value: {np.mean(episode_q_values):.3f}")
            print(f"Current Difficulty: {current_difficulty:.2f}")
            print(f"Consecutive Wins: {consecutive_wins}")
            
            if win_rate > 0.5 and current_difficulty < smartMovePlayer1:
                increase = min(0.1, smartMovePlayer1 - current_difficulty)
                current_difficulty = min(current_difficulty + increase, smartMovePlayer1)
                print(f"Increasing difficulty to: {current_difficulty:.2f}")
                consecutive_wins = 0
            
            if win_rate > 0.5 and current_difficulty == smartMovePlayer1: break
            
            agent.plot_training_metrics()
            agent.save_model()
    
    agent.save_model()
    return agent

def evaluate_agent(agent, smartMovePlayer1):
    print("Evaluating trained model")
    game = TicTacToe(smartMovePlayer1=smartMovePlayer1, playerSQN=agent)
    game.play_game()
    reward = game.get_reward()
    print(f"Final reward: {reward}")
    return reward

def main(smartMovePlayer1, mode='eval'):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
    if mode == 'train':
        if os.path.exists('2021A3PS3056G_MODEL.h5'):
            print("Pre-trained model exists.")
            agent = PlayerSQN(mode='eval')
        else:
            agent = train_agent(smartMovePlayer1)
            print("Training completed!")
            agent = PlayerSQN(mode='eval')
    else:
        agent = PlayerSQN(mode='eval')
    
    evaluate_agent(agent, smartMovePlayer1)
        
if __name__ == "__main__":
    try:
        if len(sys.argv) > 2 and sys.argv[1].lower() == 'train':
            mode = 'train'
            smartMovePlayer1 = float(sys.argv[2])
        else:
            mode = 'eval'
            smartMovePlayer1 = float(sys.argv[1])
        
        assert 0 <= smartMovePlayer1 <= 1
        
    except:
        print("Usage:")
        print("For training: python 2021A3PS3056G.py train 0.5")
        print("For evaluation: python 2021A3PS3056G.py 0.5")
        print("Note: Probability must lie between 0 and 1.")
        sys.exit(1)
    
    start_time = time()
    main(smartMovePlayer1, mode)