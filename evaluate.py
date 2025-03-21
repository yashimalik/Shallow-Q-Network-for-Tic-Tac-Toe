import importlib
import numpy as np
import os
from TicTacToe import *
from datetime import datetime

def seedGen():
    """Generates random seeds for evaluation"""
    random.seed(datetime.now().timestamp())
    rand_seed_list = []
    for i in range(100):  # Changed to 20 episodes per difficulty level
        rand_seed_list.append(random.randint(1, 1000_000_000))
    print(f"Using random seeds: {rand_seed_list}")
    return rand_seed_list

def play_full_game(playerSQN, smartness):
    """Plays a full game of TicTacToe and returns the result"""
    game = TicTacToe(smartness, playerSQN)
    game.play_game()
    reward = game.get_reward()

    if reward == 1:
        print("\033[32mWIN\033[0m")
        return 3  # 3 points for win
    elif reward == 0:
        print("\033[33mDRAW\033[0m")
        return 1  # 1 point for draw
    else:
        print("\033[31mLOSS\033[0m")
        return 0.5  # 0.5 points for loss

def evaluate_performance(submission, smartness, seeds):
    """Evaluates performance over multiple games at a given smartness level"""
    total_points = 0
    wins = 0
    draws = 0
    losses = 0
    
    print(f"\n_____________Smartness {smartness}_____________\n")
    
    # Dynamically import the module
    module_name = f"{submission}"
    my_module = importlib.import_module(module_name)
    
    for i, seed in enumerate(seeds, 1):
        random.seed(seed)
        playerSQN = my_module.PlayerSQN()
        print(f"Game {i} at smartness = {smartness}, seed = {seed}")
        
        points = play_full_game(playerSQN, smartness)
        total_points += points
        
        if points == 3:
            wins += 1
        elif points == 1:
            draws += 1
        else:
            losses += 1
            
        del playerSQN
    
    # Calculate stats
    max_possible_points = len(seeds) * 3  # games * 3 points per win
    win_rate = (wins / len(seeds)) * 100
    draw_rate = (draws / len(seeds)) * 100
    loss_rate = (losses / len(seeds)) * 100
    
    print(f"\nResults for smartness {smartness}:")
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Draws: {draws} ({draw_rate:.1f}%)")
    print(f"Losses: {losses} ({loss_rate:.1f}%)")
    print(f"Total Points: {total_points} out of {max_possible_points}")
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'total_points': total_points,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'max_possible': max_possible_points
    }

def check(submission):
    """Main evaluation function"""
    seeds = seedGen()
    results = {}
    difficulties = [0, 0.5, 1.0]
    
    print(f"\nEvaluating submission: {submission}")
    print("=" * 50)
    
    for smartness in difficulties:
        results[f'smartness_{smartness}'] = evaluate_performance(submission, smartness, seeds)
    
    # Print comprehensive results
    print("\n=== FINAL EVALUATION RESULTS ===")
    for smartness in difficulties:
        stats = results[f'smartness_{smartness}']
        print(f"\nSmartness {smartness}:")
        print(f"Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draw_rate']:.1f}%)")
        print(f"Losses: {stats['losses']} ({stats['loss_rate']:.1f}%)")
        print(f"Points: {stats['total_points']}/{stats['max_possible']}")
    
    return results

if __name__ == "__main__":
    # Get list of submissions
    current_directory = os.getcwd()
    files = [
        f
        for f in os.listdir(current_directory)
        if os.path.isfile(os.path.join(current_directory, f)) 
        and f.startswith("20") 
        and f.endswith(".py")
    ]
    
    # Store all results
    all_results = {}
    
    # Evaluate each submission
    for submission in files:
        try:
            results = check(submission[:-3])
            all_results[submission] = results
            
            # Save results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"evaluation_results_{submission[:-3]}_{timestamp}.txt"
            
            with open(results_file, 'w') as f:
                f.write(f"Evaluation Results for {submission}\n")
                f.write("=" * 50 + "\n\n")
                
                for smartness in [0, 0.5, 1.0]:
                    stats = results[f'smartness_{smartness}']
                    f.write(f"\nSmartness {smartness}:\n")
                    f.write(f"Wins: {stats['wins']} ({stats['win_rate']:.1f}%)\n")
                    f.write(f"Draws: {stats['draws']} ({stats['draw_rate']:.1f}%)\n")
                    f.write(f"Losses: {stats['losses']} ({stats['loss_rate']:.1f}%)\n")
                    f.write(f"Points: {stats['total_points']}/{stats['max_possible']}\n")
                    f.write("-" * 30 + "\n")
                
            print(f"\nResults saved to {results_file}")
            
        except Exception as e:
            print(f"\033[31mError evaluating {submission}: {str(e)}\033[0m")
            all_results[submission] = "Failed to evaluate"
    
    # Print summary of all results
    print("\n=== EVALUATION SUMMARY ===")
    for submission, results in all_results.items():
        print(f"\n{submission}:")
        if isinstance(results, dict):
            for smartness in [0, 0.5, 1.0]:
                stats = results[f'smartness_{smartness}']
                print(f"Smartness {smartness}: {stats['total_points']}/{stats['max_possible']} points "
                      f"({stats['win_rate']:.1f}% wins)")
        else:
            print(f"Status: {results}")