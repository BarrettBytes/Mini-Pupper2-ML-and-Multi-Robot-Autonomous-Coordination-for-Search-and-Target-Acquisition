Here’s an updated `README.md` that starts with downloading the `.so` file directly:


# PyKDL Setup Instructions

This repository contains the `PyKDL` shared object file (`.so`) required to use the `PyKDL` module in Python. Follow the steps below to set up `PyKDL` on your system.

## Steps to Install

1. **Download the `PyKDL` Shared Object File**

   Download the `PyKDL.cpython-310-x86_64-linux-gnu.so` file from this repository.

   - https://github.com/BarrettBytes/QuadropedRobotHonoursProject/blob/main/PyKDL/PyKDL.cpython-310-x86_64-linux-gnu.so

2. **Find Your Python User Site-Packages Directory**

   Determine the directory where your Python user-specific packages are installed by running the following command:
 

   ```bash
   python3 -m site --user-site
   ```

   This will output a directory path, typically something like:

   ```
   /home/yourusername/.local/lib/python3.x/site-packages
   ```

   Note this directory, as you will need it in the next step.

3. **Copy the `PyKDL` Shared Object File**

   Copy the downloaded `PyKDL` shared object file to the directory you found in the previous step:

   ```bash
   cp /path/to/downloaded/PyKDL.cpython-310-x86_64-linux-gnu.so /home/yourusername/.local/lib/python3.x/site-packages/
   ```

   Replace `/path/to/downloaded/` with the actual path where you saved the file, and make sure to replace `yourusername` and `python3.x` with your actual username and Python version.

    # Important
      make sure you did **NOT** just write
      cp /path/to/downloaded/PyKDL.cpython-310-x86_64-linux-gnu.so /home/yourusername/.local/lib/python3.x/site-packages/
      make sure that you actually copied the output from
      2. **Find Your Python User Site-Packages Directory**,
      and replaced
      /home/yourusername/.local/lib/python3.x/site-packages/
      with that when doing 3. **Copy the `PyKDL` Shared Object File**

4. **Verify the Installation**

   To verify that `PyKDL` is correctly installed, run the following command:

   ```bash
   python3 -c "import PyKDL; print(PyKDL.__file__)"
   ```

   If everything is set up correctly, this command will print the path to the `PyKDL` shared object file, confirming that Python can find and use the module.

you can also test with 

   ```bash
      python3 -c "import PyKDL; v = PyKDL.Vector(1, 2, 3); print(f'Vector: {v}')"
   ```
## Troubleshooting

- Ensure that you are using the correct Python version (`python3.x`) that corresponds to the `.so` file.
- If you encounter any errors, double-check the path where you copied the `.so` file to ensure it's correct.

### Explanation:

- **Step 1: Download the `.so` File**: The user is instructed to directly download the `.so` file from your GitHub repository.
- **Step 2: Find Python User Site-Packages Directory**: They find their site-packages directory using `python3 -m site --user-site`.
- **Step 3: Copy the `.so` File**: They copy the downloaded `.so` file to the identified directory.
- **Step 4: Verify the Installation**: They verify the installation by running a Python command.

Replace the link in the "Download the `.so` File" section with the actual URL of the `.so` file in your repository. This approach makes the setup process straightforward by starting with the file download.
