import os
from PIL import Image


# def convert_to_png_in_order(input_dir, output_dir, prefix):
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Get a sorted list of all files in the input directory
#     image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')])

#     # Loop through the sorted list and convert to PNG
#     for i, filename in enumerate(image_files, start=1):
#         # Construct the full file path
#         file_path = os.path.join(input_dir, filename)
        
#         # Open the image file
#         try:
#             with Image.open(file_path) as img:
#                 # Create the output file path with the format prefix_ith_image.png
#                 output_file = f"{prefix}_{i}.png"
#                 output_path = os.path.join(output_dir, output_file)
                
#                 # Save the image as PNG
#                 img.save(output_path, 'PNG')
#                 print(f"Converted {filename} to {output_file}")
#         except Exception as e:
#             print(f"Failed to convert {filename}: {e}")

# This version maintains the image base name after the conversion
def convert_to_png_in_order(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a sorted list of all files in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png','.PNG','.JPG','.jpeg','.JPEG', '.jpg', 'webp'))])

    # Loop through the sorted list and convert to PNG
    for i, filename in enumerate(image_files, start=1):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)
        image_base_name = os.path.splitext(filename)[0] # Gets image base name without extention
        
        # Open the image file
        try:
            with Image.open(file_path) as img:
                # Create the output file path with the format prefix_ith_image.png
                output_file = f"{image_base_name}.png"
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

def remove_end_of_filename(input_dir):
    # Removes additional ending of file name 
    
    # Set to track processed pairs
    processed_files = set()

    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png','.PNG','.JPG','.jpeg','.JPEG', '.jpg'))]

    # Process each image 
    for i, image_filename in enumerate(image_files, start=1):
        image_base_name = os.path.splitext(image_filename)[0]  # Get the base name without extension

        # Check if the image file has already been processed
        if image_base_name in processed_files:
            continue  # Skip if already processed
        
        # Constructs old and new file paths
        old_image_path = os.path.join(input_dir, image_filename)
        image_base_name = image_base_name.rsplit(' ', 1)[0] # Removes the end string of file name seperated by a space
        new_image_name = f"{image_base_name}{os.path.splitext(image_filename)[1]}"  # Retain original extension
        new_image_path = os.path.join(input_dir, new_image_name)


        # Rename the image and txt file
        try:
            os.rename(old_image_path, new_image_path)
            print(f"Renamed {image_filename} to {new_image_name}")

            # Add base names of processed image and txt file to the set
            processed_files.add(image_base_name)

        except Exception as e:
            print(f"Failed to rename files {image_filename}: {e}")    


def rename_image_txt_pairs(input_dir, prefix, start_num=0):
    """
    Renames all image and txt file pairs in the input directory to a specified prefix.
    
    Args:
    - input_dir (str): Path to the directory containing images and txt files.
    - prefix (str): The new prefix to use for renaming image and txt files.
    """
    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png','.PNG','.JPG','.jpeg','.JPEG', '.jpg'))]

    # Set to track processed pairs
    processed_files = set()

    # Process each image and find its corresponding txt file
    for i, image_filename in enumerate(image_files, start=1):
        image_base_name = os.path.splitext(image_filename)[0]  # Get the base name without extension

        # Check if the image file has already been processed
        if image_base_name in processed_files:
            continue  # Skip if already processed

        # Search for a corresponding .txt file with the same base name
        txt_filename = f"{image_base_name}.txt"
        txt_path = os.path.join(input_dir, txt_filename)

        # If no matching .txt file is found, continue to search the directory
        if not os.path.exists(txt_path):
            print(f"Warning: No corresponding .txt file found for {image_filename}. Skipping.")
            continue

        # New names for the image and txt files
        new_image_name = f"{prefix}_{i+start_num}{os.path.splitext(image_filename)[1]}"  # Retain original extension
        new_txt_name = f"{prefix}_{i+start_num}.txt"

        # Construct full old and new file paths
        old_image_path = os.path.join(input_dir, image_filename)
        old_txt_path = os.path.join(input_dir, txt_filename)
        new_image_path = os.path.join(input_dir, new_image_name)
        new_txt_path = os.path.join(input_dir, new_txt_name)

        # Rename the image and txt file
        try:
            os.rename(old_image_path, new_image_path)
            os.rename(old_txt_path, new_txt_path)
            print(f"Renamed {image_filename} to {new_image_name} and {txt_filename} to {new_txt_name}")

            # Add base names of processed image and txt file to the set
            processed_files.add(image_base_name)

        except Exception as e:
            print(f"Failed to rename files {image_filename} and {txt_filename}: {e}")


import os

def replace_any_string_in_txt_files(directory, replacement_digit):
    """
    Goes through a directory and replaces the first word (string) of each line in all .txt files
    with the specified digit.
    
    Args:
    - directory (str): The path to the directory containing the .txt files.
    - replacement_digit (str or int): The digit to replace the first string with.
    """
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Process each txt file
    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)

        # Read the contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Replace the first word (string) of each line with the replacement digit
        with open(file_path, 'w') as file:
            for line in lines:
                # Split the line into words (assumes words are separated by spaces)
                parts = line.split(maxsplit=1)

                if parts:
                    # Replace the first word (string) with the replacement digit
                    new_line = f"{replacement_digit} {parts[1]}" if len(parts) > 1 else f"{replacement_digit}\n"
                    file.write(new_line)
                else:
                    # In case the line is empty, write it as is
                    file.write(line)

        print(f"Processed {txt_file}")



if __name__ == '__main__':
    ######################### FUNCTION 1 ############################
    #input_dir = "/home/eherrin@ad.ufl.edu/Documents/test4_251_DK_ugv"
    # output_dir = "/home/eherrin@ad.ufl.edu/Documents/20241003_au_park_jackal/images_converted"
    # prefix = "jackal_single"
    #convert_to_png_in_order(input_dir, output_dir, prefix)
    #rename_txt_files_in_order(input_dir, output_dir, prefix)

    ######################### FUNCTION 2 ############################
    # Use the function to update the files in the given directory
    # directory = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/drone_dataset_v2/valid/labels"
    #directory = "/home/eherrin@ad.ufl.edu/Documents/labels"


    ######################### FUNCTION 3 ############################
    # input_dir = "/home/eherrin@ad.ufl.edu/Documents/test4_251_DK"
    # delete_lines_starting_with(input_dir, element_to_remove=1) # making UGV

    ########################### COMBINED FUNCTIONS ######################
    '''
    Rename all class labels to txt's from n to class 1, then rename all images with a new prefix, and do the same for the txt's

    '''
    input_dir = "/home/eherrin@ad.ufl.edu/Documents/t5_clutter_ugv/train_yolo/spot"
    #output_dir = "/home/eherrin@ad.ufl.edu/Documents/test_8_jakal_new/images/converted_to_png"
    #input_dir = "/home/eherrin@ad.ufl.edu/Documents/ugv_car_counterexamples_train/labels"
    # prefix = "RAITE_jackal_dell"
    #rename_image_txt_pairs(input_dir, prefix)
    # find_the_empty_txt(input_dir)
    # replace_first_element_in_files(input_dir, old_value=0, new_value=1)
    # replace_first_element_in_files(input_dir, old_value=2, new_value=0)
    # replace_first_element_in_files(input_dir, old_value=3, new_value=0)
    # replace_first_element_in_files(input_dir, old_value=4, new_value=0)
    replace_first_element_in_files(input_dir, old_value=1, new_value=4)
    # replace_any_string_in_txt_files(input_dir, 0)
    # convert_to_png_in_order(input_dir, output_dir, prefix)
    # rename_image_txt_pairs(input_dir, prefix)
    # remove_end_of_filename(input_dir)



