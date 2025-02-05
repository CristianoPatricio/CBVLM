import os

def find_leaf_dirs_without_results(directory):
    # List to hold leaf directories without results.json
    leaf_dirs_without_results = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        if len(dirs) == 1 and ".hydra" in dirs[0] and not root.endswith(".hydra"):
            if 'results.json' not in files:
                leaf_dirs_without_results.append(root)
                
    return leaf_dirs_without_results

# Example usage:
directory = '/home/icrto/multimodal-LLM-explainability-dev/logs/multiruns'
missing_results = find_leaf_dirs_without_results(directory)

# Print the leaf directories without results.json
for subdir in missing_results:
    print(subdir)
