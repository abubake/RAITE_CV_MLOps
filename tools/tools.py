import os
from PIL import Image

def convert_to_png(input_dir, output_dir, prefix):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for i, filename in enumerate(os.listdir(input_dir), start=1):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)
        
        # Open the image file
        try:
            with Image.open(file_path) as img:
                # Create the output file path with the format prefix_ith_image.png
                output_file = f"{prefix}_{i}.png"
                output_path = os.path.join(output_dir, output_file)
                
                # Save the image as PNG
                img.save(output_path, 'PNG')
                print(f"Converted {filename} to {output_file}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")


def replace_first_element_in_files(directory, old_value=2, new_value=1):
    # Loop through all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the first element of each line
            modified_lines = []
            for line in lines:
                elements = line.split()
                if len(elements) > 0 and elements[0] == str(old_value):
                    elements[0] = str(new_value)
                modified_lines.append(' '.join(elements) + '\n')

            # Write the modified lines back to the same file
            with open(file_path, 'w') as file:
                file.writelines(modified_lines)

            print(f"Updated {filename}")
    

def delete_lines_starting_with(directory, element_to_remove):
    # Loop through all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Filter out lines that start with the given element
            filtered_lines = []
            for line in lines:
                elements = line.split()
                if len(elements) > 0 and elements[0] != str(element_to_remove):
                    filtered_lines.append(line)

            # Write the filtered lines back to the same file
            with open(file_path, 'w') as file:
                file.writelines(filtered_lines)

            print(f"Updated {filename} (removed lines starting with {element_to_remove})")



if __name__ == '__main__':
    ######################### FUNCTION 1 ############################
    # input_dir = "/home/eherrin@ad.ufl.edu/Documents/ugvs"
    # output_dir = "/home/eherrin@ad.ufl.edu/Documents/ugvs_converted"
    # prefix = "ugv_image"
    # convert_to_png(input_dir, output_dir, prefix)

    ######################### FUNCTION 2 ############################
    # Use the function to update the files in the given directory
    directory = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/ugv_dataset/train/labels"
    replace_first_element_in_files(directory)

    ######################### FUNCTION 3 ############################
    # directory = "/home/eherrin@ad.ufl.edu/Downloads/lab_robots_video_labels"
    # delete_lines_starting_with(directory, element_to_remove=2) # making UGV