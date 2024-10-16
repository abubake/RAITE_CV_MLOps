import os
from PIL import Image

def convert_to_png_in_order(input_dir, output_dir, prefix):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a sorted list of all files in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # Loop through the sorted list and convert to PNG
    for i, filename in enumerate(image_files, start=1):
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


def rename_txt_files_in_order(input_dir, output_dir, prefix):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a sorted list of all .txt files in the input directory
    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    # Loop through the sorted list and rename the .txt files
    for i, filename in enumerate(txt_files, start=1):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)

        # Create the new file name with the format prefix_ith_file.txt
        new_filename = f"{prefix}_{i}.txt"
        output_path = os.path.join(output_dir, new_filename)

        # Copy and rename the .txt file
        try:
            # Read the contents of the original file
            with open(file_path, 'r') as file:
                content = file.read()

            # Write the contents to the new file
            with open(output_path, 'w') as new_file:
                new_file.write(content)

            print(f"Renamed {filename} to {new_filename}")
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")


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

def find_the_empty_txt(directory):
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Check if it's a .txt file and if it's empty
        if filename.endswith('.txt') and os.path.getsize(file_path) == 0:
            print(f"Empty file: {filename}")

def convert_txt_indexes_and_rename_images_and_txts(input_dir, output_dir, prefix): #, original_class=2, output_class=1):

    #replace_first_element_in_files(input_dir, old_value=original_class, new_value=output_class)
    convert_to_png_in_order(input_dir, output_dir, prefix)
    rename_txt_files_in_order(input_dir, output_dir, prefix)


if __name__ == '__main__':
    ######################### FUNCTION 1 ############################
    input_dir = "/home/eherrin@ad.ufl.edu/Documents/test4_251_DK_ugv"
    # output_dir = "/home/eherrin@ad.ufl.edu/Documents/20241003_au_park_jackal/images_converted"
    # prefix = "jackal_single"
    #convert_to_png_in_order(input_dir, output_dir, prefix)
    #rename_txt_files_in_order(input_dir, output_dir, prefix)

    ######################### FUNCTION 2 ############################
    # Use the function to update the files in the given directory
    # directory = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/drone_dataset_v2/valid/labels"
    #directory = "/home/eherrin@ad.ufl.edu/Documents/labels"
    # replace_first_element_in_files(input_dir, old_value=2, new_value=1)


    ######################### FUNCTION 3 ############################
    # input_dir = "/home/eherrin@ad.ufl.edu/Documents/test4_251_DK"
    # delete_lines_starting_with(input_dir, element_to_remove=1) # making UGV

    ########################### COMBINED FUNCTIONS ######################
    '''
    Rename all class labels to txt's from n to class 1, then rename all images with a new prefix, and do the same for the txt's

    '''
    input_dir = "/home/eherrin@ad.ufl.edu/Documents/20241003_autonomy_park_human_and_ugvs5_data/labels"
    output_dir = "/home/eherrin@ad.ufl.edu/Documents/20241003_autonomy_park_human_and_ugvs5_data/labels/converted"
    prefix = "large_spot_short_clip"
    #convert_txt_indexes_and_rename_images_and_txts(input_dir, output_dir, prefix)
    #convert_to_png_in_order(input_dir, output_dir, prefix)
    rename_txt_files_in_order(input_dir, output_dir, prefix)
