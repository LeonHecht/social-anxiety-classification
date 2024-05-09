import os
import json
from transformers import TrainingArguments

def find_checkpoints_with_metric_and_args(base_path, target_metric):
    """
    Searches through checkpoint directories within a base path for trainer_state.json files
    with a best metric value approximately equal to the target and reads the training arguments.

    Parameters:
        base_path (str): The directory containing the checkpoint folders.
        target_metric (float): The target metric value to search for.

    Returns:
        dict: A dictionary where keys are directories and values are the training arguments.
    """
    results = {}
    # Traverse the base directory
    for dirname in os.listdir(base_path):
        dir_path = os.path.join(base_path, dirname)
        if os.path.isdir(dir_path):
            state_file_path = os.path.join(dir_path, "trainer_state.json")
            args_file_path = os.path.join(dir_path, "training_args.bin")
            if os.path.exists(state_file_path) and os.path.exists(args_file_path):
                with open(state_file_path, 'r') as file:
                    state_data = json.load(file)
                    # Check if the best_metric key exists and matches the target metric
                    if 'best_metric' in state_data and round(state_data['best_metric'], 2) == round(target_metric, 2):
                        # Load the training arguments using transformers
                        training_args = TrainingArguments.from_json_file(args_file_path)
                        results[dir_path] = training_args

    return results

# Example usage:
base_directory = "results/"
metric_of_interest = 0.68
matching_checkpoints_and_args = find_checkpoints_with_metric_and_args(base_directory, metric_of_interest)
print("Checkpoints and their training arguments matching the metric of interest:")
for path, args in matching_checkpoints_and_args.items():
    print(f"Path: {path}")
    print(f"Arguments: {args}")
