import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def read_tensorboard_events(event_file):
    """
    Read TensorBoard event file and extract Train and Eval average returns.
    
    Args:
        event_file: Path to the TensorFlow events file
        
    Returns:
        Dictionary containing train and eval data with iterations and returns
    """
    train_returns = []
    eval_returns = []
    train_iterations = []
    eval_iterations = []
    
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag == 'Train_AverageReturn':
                train_returns.append(value.simple_value)
                train_iterations.append(event.step)
            elif value.tag == 'Eval_AverageReturn':
                eval_returns.append(value.simple_value)
                eval_iterations.append(event.step)
    
    return {
        'train_returns': train_returns,
        'train_iterations': train_iterations,
        'eval_returns': eval_returns,
        'eval_iterations': eval_iterations
    }


def plot_eval_returns(data, save_path='eval_average_return_plot.png'):
    """
    Plot Eval Average Return vs. number of iterations.
    
    Args:
        data: Dictionary containing eval_returns and eval_iterations
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if data['eval_returns']:
        plt.plot(data['eval_iterations'], data['eval_returns'], 
                marker='o', linewidth=2, markersize=8, 
                label='Eval AverageReturn', color='blue')
    
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Eval Average Return', fontsize=12)
    plt.title('Eval Average Return vs. Number of Iterations', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_path}'")
    plt.show()


def main():
    # Path to your TensorFlow events file
    event_file = "/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw4/rob831/data/hw4_q3_reacher_reacher-hw4_part1-v0_09-11-2025_11-54-07/events.out.tfevents.1762707247.Esmes-Air-5"
    
    # Read the data
    data = read_tensorboard_events(event_file)
    
    # Print extracted data
    print(f"Found {len(data['train_returns'])} Train_AverageReturn values: {data['train_returns']}")
    print(f"Found {len(data['eval_returns'])} Eval_AverageReturn values: {data['eval_returns']}")
    
    # Plot eval returns
    plot_eval_returns(data)


if __name__ == "__main__":
    main()