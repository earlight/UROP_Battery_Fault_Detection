# Dataset Source/Availability

Dataset from the paper: https://www.nature.com/articles/s41467-023-41226-5

Link: https://doi.org/10.6084/m9.figshare.23659323

Backup Link: https://figshare.com/articles/dataset/Realistic_fault_detection_of_Li-ion_battery_via_dynamical_deep_learning_approach/23659323

The dataset comes in the form of many pkl files (690,000 pkl files accross 3 car brands). 

# Instructions to convert data into CSV

1. Download dataset (containing all 3 brands) from the links above.

2. Move dataset into repo

3. You might have to change the DATASET_PATH in process_data.py to point to the main dataset folder that contains all 3 brands.

3. Run process_data.py, which converts the 690,000 pkl files into 3 CSV files, one for each brand. process_data.py took around 15 minutes to run on my M2 Macbook, but the time may vary depending on your specs. The 3 CSV files should be around 8.97 GB in size.