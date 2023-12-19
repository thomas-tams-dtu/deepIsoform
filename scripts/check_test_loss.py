#!/usr/bin/env python3

import os
import re

def extract_part_between_ts_and_underscore(file_name):
    match = re.search(r'tl(.*)_?', file_name)
    if match:
        return match.group(1)
    else:
        return None

def check_better_test_loss(model_test_loss, model_prefix, model_dir):
    try:
        files = os.listdir(model_dir)
        files_list = [file for file in files if file.startswith(model_prefix)]

        if not files_list:
            print(f"No files starting with {model_prefix} found in the folder.")
            return True
        else:
            for file in files_list:
                extracted_part = extract_part_between_ts_and_underscore(file)
                if extracted_part is not None:
                    print(f"File: {file}, Extracted part: {extracted_part}")
                    
                    if model_test_loss < float(extracted_part):
                        print(f"New model performs better. Removing {file}")
                        os.remove(f"{model_dir}/{file}")
                        return True
                    else:
                        print(f"New model performs worse")
                        return False

    except FileNotFoundError:
        print(f"Folder '{model_dir}' not found.")