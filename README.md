# Chlamy-IMPI

This repository contains the image processing and data analysis pipeline used to study photosynthesis in a large
number of Chlamydomonas reinhardtii mutants under various growth conditions.

Note: this repository is still under development, so things might break and these instructions might be incomplete.


## Image processing

###: 0. How to run script covering entire pipeline:

How to run script on lab windows laptop:

    1. Windows key. Type anaconda powershell. Select option: run as administrator.
    2. Type: cd “C:\Users\Burlacot Lab\Documents\GitHub\chlamy-ImPi\scripts” (quotes are important here)
    3. Type: .\generate_database.ps1
    4. Monitor output from code
    5. New files will appear in C:\Users\Burlacot Lab\Documents\GitHub\chlamy-ImPi\output\database_creation

### 1. Well segmentation

After setting up the environment (see below), download all image data from the google drive folder,
located at: https://drive.google.com/drive/folders/1rU8VOIdwBuDX_N6MTn0Bg5SYYb-Ov8zv and place all .tif files
in the `data` directory in this project. Then run the image processing pipeline:

```
python well_segmentation_preprocessing/main.py
```

This should store a bunch of `.npy` files in the `output/well_segmentation_cache` directory. Note- you can skip this step, and download 
the `.npy` files directly from a shared cache here: https://drive.google.com/drive/folders/1LB1znkc95zbgKAPVU2Rz4MMwbdcjtsBK


### 2. Database creation

After running well_segmentation_preprocessing/main.py with new data, we need to first manually correct the 
mysterious dodgy time points which are intermittently present in the data. I couldn't easily find a 
way to automate this step which worked every time.

Run:

```
python database_creation/investigate_meta_df.py
```

and manually edit error_correction.py:manually_fix_erroneous_time_points to fix them.

Now we can create the database:

```
python database_creation/main.py
```


Note: see the file `paths.py` for where the code looks to read and write these file caches from.


## Data analysis

Coming soon.


## Poetry instructions:

First install poetry: https://python-poetry.org/docs/#installation

Next, install the dependencies and activate the virtual environment:

```
$ poetry install
$ poetry shell
```

## Streamlit instructions:

```
$ poetry shell
$ streamlit run chlamy_impi/interactive.demo.py
```
