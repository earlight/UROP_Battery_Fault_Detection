import numpy as np
import pandas as pd
import time
import torch
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

start_time = time.time()

DATASET_PATH = 'raw_data/'
BRAND = 1

# load column names
columns = torch.load(f'{DATASET_PATH}battery_brand{BRAND}/column.pkl')
columns.extend(['label', 'car', 'charge_segment', 'mileage'])

# get folders for each brand
folders = []
if BRAND == 3:
    folders.append(f'{DATASET_PATH}battery_brand{BRAND}/data')
else:
    folders.append(f'{DATASET_PATH}battery_brand{BRAND}/train')
    folders.append(f'{DATASET_PATH}battery_brand{BRAND}/test')

# store data for each car
car_data = defaultdict(list)

# open pkl files in all folders
for folder in folders:
    print("Processing folder:", folder)

    for test_pkl in tqdm(listdir(folder)):
        time_series_data, metadata = torch.load(join(folder, test_pkl))
        
        time_series_data = time_series_data.tolist()

        # add metadata
        label, car, charge_segment, mileage = metadata.values()
        for i in range(len(time_series_data)):
            time_series_data[i].extend((int(label[:1]), car, charge_segment, mileage))

        # add to car_data
        car_data[car].extend(time_series_data)

# store all data to one csv file
brand_df = pd.concat([pd.DataFrame(car_data[car], columns=columns) for car in tqdm(car_data.keys())])
print("Saved brand data to dataframe. Shape:", brand_df.shape)

# save to csv
csv_start_time = time.time()
brand_df.to_csv(f'data/brand{BRAND}.csv', index=False)
print("Brand", BRAND, "processed. Shape:", brand_df.shape)
print("Time to save csv:", time.time() - csv_start_time, "seconds\n")

print(f'--- {time.time() - start_time} seconds ---')