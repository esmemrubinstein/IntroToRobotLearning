from read_results import get_section_results
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def load_experiments(base_dir, prefix):
    """Loads all runs matching a prefix, e.g. 'q1_dqn_' or 'q1_doubledqn_'."""
    runs = []
    for folder in sorted(os.listdir(base_dir)):
        if folder.startswith(prefix):
            path = os.path.join(base_dir, folder)
            event_files = glob.glob(os.path.join(path, 'events*'))
            if not event_files:
                print(f"Skipping {folder}: no event file found")
                continue
            X, Y = get_section_results(event_files[0])
            runs.append((X, Y))
            print(f"Loaded {folder} ({len(X)} points)")
    if not runs:
        raise RuntimeError(f"No valid runs found for prefix '{prefix}'")
    return runs


def average_runs(runs):
    avgY = (np.array(runs[0][1]) + np.array(runs[1][1]) + np.array(runs[2][1]))/3
    X = runs[0][0]
    min_Y = np.min((np.array(runs[0][1]), np.array(runs[1][1]), np.array(runs[2][1])), axis=0)
    max_Y = np.max((np.array(runs[0][1]), np.array(runs[1][1]), np.array(runs[2][1])), axis=0)

    return avgY, X, min_Y, max_Y


def plot_dqn():
    base_dir = '/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw3/data'
    for label, prefix in [("DQN", "q1_dqn_"), ("Double DQN", "q1_doubledqn_")]:
        runs = load_experiments(base_dir, prefix)
        avg_Y, X, min_Y, max_Y= average_runs(runs)
        (line,) = plt.plot(X, [None] + list(avg_Y), label=label)
        color = line.get_color()
        plt.fill_between(X[1:], min_Y, max_Y,
                 color=color, alpha=0.15)

    plt.xlabel('Train Iteration')
    plt.ylabel('Train Return')
    plt.legend()
    plt.title('Deep Q network vs Double Deep Q network')
    plt.show()


def plot_learning_curve():
    base_dir = '/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw3/data'
    for label, prefix in [("ActorCritic-Q3", "q3_10_10_InvertedPend")]:
        runs = load_experiments(base_dir, prefix)
        plt.plot(runs[0][0], runs[0][1], label=label)
    plt.xlabel('Train Iteration')
    plt.ylabel('Eval Return')
    plt.title("Actor Critic Cartpole-v0")
    plt.show()

plot_learning_curve()

        

