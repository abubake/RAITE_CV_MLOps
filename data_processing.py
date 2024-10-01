import os

# Path to the folder containing the label files
labels_dir = 'data/archive/drone_dataset/train/labels'

# Iterate over all files in the labels directory
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(labels_dir, filename)
        
        # Read the current contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Update the first element of each line from 0 to 1
        updated_lines = []
        for line in lines:
            parts = line.split()
            if parts[0] == '0':  # Check if the first element is 0
                parts[0] = '1'  # Change it to 1
            updated_lines.append(' '.join(parts))
        
        # Write the updated contents back to the file
        with open(file_path, 'w') as file:
            file.write('\n'.join(updated_lines))

print("Label files updated successfully.")
