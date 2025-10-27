import os
import glob
import matplotlib.pyplot as plt
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to your data folder
data_dir = "/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw2/data"

# Prefix for the experiments we want
prefix = "parallel"

# Find all event files for experiments that start with q1_sb_
event_files = glob.glob(os.path.join(data_dir, prefix + "*/events.out.tfevents.*"))
print(event_files)

def load_event_data(event_file, tag="Eval_AverageReturn"):
    """Load a scalar tag (like Eval_AverageReturn) from a TensorBoard event file."""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    if tag not in event_acc.Tags()["scalars"]:
        raise ValueError(f"Tag {tag} not found in {event_file}")
    scalar_events = event_acc.Scalars(tag)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    return steps, values

def clean_exp_name(folder_name: str) -> str:
    # Remove timestamp-like numbers (e.g., 26-09-2025, 12-15-01)
    no_nums = re.sub(r'[_-]?\d+', '', folder_name)
    # Remove last 2 values
    no_nums = no_nums[:-2]
    return no_nums

# Collect results first
results = []

for event_file in event_files:
    folder_name = os.path.basename(os.path.dirname(event_file))
    exp_name = folder_name # clean_exp_name(folder_name)
    
    steps, returns = load_event_data(event_file, tag="Eval_AverageReturn")
    results.append((exp_name, steps, returns))

# Sort results alphabetically by experiment name
results.sort(key=lambda x: x[0])

# Plot after sorting
plt.figure(figsize=(10, 6))
for exp_name, steps, returns in results:
    plt.plot(steps, returns, label=exp_name)

plt.xlabel("Iteration (Envsteps)")
plt.ylabel("Average Return")
plt.title("Compare Parallel Lunar Lander vs Non Parallel")
plt.legend()
plt.grid(True)
plt.show()
