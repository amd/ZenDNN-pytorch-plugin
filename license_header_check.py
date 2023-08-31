# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import re
import glob

parent_dir = os.getcwd()  # not using abspath (req. for windows) here
all_files_lst = sorted(
    glob.glob(f"{parent_dir}/src/**/*.cpp", recursive=True)
    + glob.glob(f"{parent_dir}/src/**/*.hpp", recursive=True)
    + glob.glob(f"{parent_dir}/src/**/*.py", recursive=True)
    + glob.glob(f"{parent_dir}/test/**/*.py", recursive=True)
    + glob.glob(f"{parent_dir}/cmake/**/*.cmake", recursive=True)
)

files_to_upd_lst = []

for file in all_files_lst:
    with open(file, "r") as f:
        file_str = f.read()
        # use re.DOTALL and re.IGNORECASE flags (bitwise OR needed)
        match_obj = re.search(
            "Copyright.*Advanced Micro Devices.*All rights reserved",
            file_str,
            re.S | re.I,
        )
        if match_obj is None:
            files_to_upd_lst.append(file)

if len(files_to_upd_lst) > 0:
    print("The following files are missing the copyright headers, please add them!")
    for file in files_to_upd_lst:
        print(file)
else:
    print("License header is present in all the files!")

print("License check completed!")
