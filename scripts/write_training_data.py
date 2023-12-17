import os

def write_training_data(file_path, metadata_dict):
    
    # Check if file exists
    print(os.path.exists(file_path))
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            header_list = metadata_dict.keys()

            header_string = '\t'.join(map(str, header_list)) + '\n'
            file.write(header_string)
    
    # Read the existing content
    with open(file_path, 'r') as file:
        existing_content = file.read()

    # Add a line to the existing content
    metadata_list = metadata_dict.values()
    line_to_add = '\t'.join(map(str, metadata_list)) + '\n'
    new_content = existing_content + line_to_add

    # Write the new content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
