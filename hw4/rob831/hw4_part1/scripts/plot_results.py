import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Path to your TensorFlow events file
event_file = "/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw4/rob831/data/hw4_q2_obstacles_singleiteration_obstacles-hw4_part1-v0_10-11-2025_12-50-50/events.out.tfevents.1762797050.Esmes-Air-5"

# Initialize lists to store data
train_returns = []
eval_returns = []
train_iterations = []
eval_iterations = []

# Read the event file
for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        if value.tag == 'Train_AverageReturn':
            train_returns.append(value.simple_value)
            train_iterations.append(event.step)
        elif value.tag == 'Eval_AverageReturn':
            eval_returns.append(value.simple_value)
            eval_iterations.append(event.step)

# Print extracted data
print(f"Found {len(train_returns)} Train_AverageReturn values: {train_returns}")
print(f"Found {len(eval_returns)} Eval_AverageReturn values: {eval_returns}")

# Create the plot
plt.figure(figsize=(10, 6))

if train_returns:
    plt.scatter(train_iterations, train_returns, s=100, marker='o', 
                label='Train AverageReturn', color='blue', zorder=3)
    
if eval_returns:
    plt.scatter(eval_iterations, eval_returns, s=100, marker='s', 
                label='Eval AverageReturn', color='red', zorder=3)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Average Return', fontsize=12)
plt.title('Train and Eval Average Returns', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Set x-axis to show iteration 0 clearly
if train_iterations or eval_iterations:
    all_iters = train_iterations + eval_iterations
    plt.xlim(-0.5, max(all_iters) + 0.5)

plt.tight_layout()
plt.savefig('average_returns_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'average_returns_plot.png'")
plt.show()