import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Base directory containing all run folders
base_dir = "/Users/esmerubinstein/Desktop/IntroToRobotLearning/hw4/rob831/data"

# Keywords to look for in folder names
params = ["cem_4","cem_2", "random"]

# Collect folders that contain any of the keywords
matching_folders = [
    os.path.join(base_dir, d)
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and any(p in d for p in params)
]

if not matching_folders:
    print("⚠️ No folders found containing any of:", params)
else:
    plt.figure(figsize=(8, 6))

    for folder in matching_folders:
        # Find TensorBoard event files
        event_files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))
        if not event_files:
            print(f"⚠️ No event file found in {folder}")
            continue
        
        event_path = event_files[0]
        
        # Extract name based on keyword found in folder
        run_name = None
        for p in params:
            if p in folder:
                run_name = [part for part in folder.split("_") if p in part]
                run_name = run_name[0] if run_name else p
                break
        
        # Load TensorBoard data
        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()
        
        tag = "Eval_AverageReturn"
        if tag not in ea.Tags().get("scalars", []):
            print(f"⚠️ Tag '{tag}' not found in {folder}")
            continue

        scalar_events = ea.Scalars(tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        
        plt.plot(steps, values, label=run_name)

    # Plot formatting
    plt.title("Evaluation Average Return vs Training Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Eval Average Return")
    plt.legend(title="Run")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cem_random_eval_average_return_comparison.png")
    plt.show()


