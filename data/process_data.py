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

# process data for each brand
for brand in range(1,4):

    # load column names
    columns = torch.load(f'{DATASET_PATH}battery_brand{brand}/columns.pkl')
    columns.extend(['label', 'car', 'mileage'])

    # get folders for each brand
    folders = []
    if brand == 3:
        folders.append(f'{DATASET_PATH}battery_brand{brand}/train')
    else:
        folders.append(f'{DATASET_PATH}battery_brand{brand}/train')
        folders.append(f'{DATASET_PATH}battery_brand{brand}/test')

    # store data for each car
    car_data = defaultdict(list)

    # open pkl files in all folders
    for folder in folders:
        print("Processing folder:", folder)

        # open all pkl files in folder
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        print("Number of pkl files:", len(files))

        for test_pkl in tqdm(files):
            time_series_data, metadata = torch.load(join(folder, test_pkl))
            time_series_data = time_series_data.tolist()

            # add metadata
            label, car, charge_segment, mileage = metadata.values()
            for i in range(len(time_series_data)):
                time_series_data[i].extend((int(label[:1]), car, mileage))

            # add to car_data
            car_data[car].extend(time_series_data)

    # store all data to one csv file
    dataframe_start_time = time.time()
    brand_df = pd.concat([pd.DataFrame(car_data[car], columns=columns) for car in car_data.keys()])
    print("Time to create dataframe:", time.time() - dataframe_start_time, "seconds")

    # save to csv
    csv_start_time = time.time()
    brand_df.to_csv(f'processed_data/brand{brand}.csv', index=False)
    print("Brand", brand, "processed. Shape:", brand_df.shape)
    print("Time to save csv:", time.time() - csv_start_time, "seconds")

print(f'--- {time.time() - start_time} seconds ---')