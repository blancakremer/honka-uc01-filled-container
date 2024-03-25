"""
This module defines the data_processing function used by the pipeline orchestrator to perform data preprocessing. 
This function defines the logic for data preprocessing. Any adidtional function needed to perform this step can 
be defined within this script itself or split into different scripts and included in the Process directory.
"""

from sklearn.model_selection import train_test_split
from typing import Dict, Any
import pandas as pd
import datetime
import numpy as np
def process_core(necessary_data_columns_frame):
    x_tag_list = ["SH","Category","Weight","Filling start"]  # source, material type, weight, start time
    x_array = necessary_data_columns_frame[x_tag_list].values.tolist()
    end_time_array = necessary_data_columns_frame["Filling end"].tolist()  # end time
    loading_time_array = necessary_data_columns_frame["Pickup"].tolist()
    #print(end_time_array)
    #source_array = necessary_data_columns_frame["Unnamed: 2"].tolist()
    #mat_list = necessary_data_columns_frame["Unnamed: 3"].tolist()
    #print(x_array)
    #print(end_time_array)
    #print(loading_time_array)
    # sometimes the start filling ts are changed
    alt_start_ts = necessary_data_columns_frame["Alt filling start"]
    alt_start_ts.fillna("", inplace=True)
    alt_start_ts = alt_start_ts.tolist()
    trip_tag_array = necessary_data_columns_frame["trip_id"].tolist()
    #print(trip_tag_array)
    #license plate
    ls_plate_array = necessary_data_columns_frame["Plate"].tolist()
    #arrival to factory time
    arrival_array = necessary_data_columns_frame["Arrival"].tolist()

    source_tag = {"sh1":1,
                  "sh2":2,
                  "sh3":3}
    mat_tag = {"Sekatuote siipikarja luokka 3":1,
               "Varpaat, Siipikarja, Luokka 3":2}



    x_array_processed = []
    y_array_processed = []
    trip_tag_array_processed = []
    total_loading_wait_sec=[]
    counter =0
    counter_anom = 0
    anom_count = 0
    for idx in range(len(x_array)):
        try:
            #process source to tag num

            #process start time to seconds
            if alt_start_ts[idx] != "":
                start_timestamp_str = alt_start_ts[idx]
            else:
                start_timestamp_str = x_array[idx][3]
            start_timestamp = datetime.datetime.strptime(start_timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            week_day = start_timestamp.weekday()

            end_timestamp = datetime.datetime.strptime(end_time_array[idx], '%Y-%m-%dT%H:%M:%SZ')

            loading_wait_delta = datetime.datetime.strptime(loading_time_array[idx],
                                                            '%Y-%m-%dT%H:%M:%SZ') - end_timestamp

            weight_float = float("".join(x_array[idx][2][:-2]))

            #process time spent to seconds

            total_second_spent = (end_timestamp-start_timestamp).total_seconds()
            if total_second_spent < 300 or total_second_spent > 80000:
                anom_count +=1
                continue
            if total_second_spent > 0 and total_second_spent < 60:
                counter+=1
            week_no = datetime.date(year=start_timestamp.year, month=start_timestamp.month,
                                    day=start_timestamp.day).isocalendar()[1]
            # 3 input variables structures
            # x_array_processed.append([day_of_year,weight_float,week_day,source_tag[x_array[idx][0]], mat_tag[x_array[idx][1]], total_sec_from_begining_of_day, start_timestamp.year, start_timestamp.month,start_timestamp.day]) #with weight
            # x_array_processed.append([day_of_year, week_day, source_tag[x_array[idx][0]], mat_tag[x_array[idx][1]],total_sec_from_begining_of_day, int(year)]) #without weight
            x_array_processed.append([week_day, source_tag[x_array[idx][0]], mat_tag[x_array[idx][1]], start_timestamp.month, start_timestamp.day, start_timestamp.hour, start_timestamp.minute, week_no]) # without weight, detail start timestamp
            y_array_processed.append(np.log10(total_second_spent))

            if loading_wait_delta.total_seconds() < -1:
                counter_anom+=1
                print(loading_time_array[idx])
            total_loading_wait_sec.append(loading_wait_delta.total_seconds())
            trip_tag_array_processed.append(trip_tag_array[idx])

        except Exception as err:
            print(err, " happened at ",err.__traceback__.tb_lineno)
            #odd, most if not all filling end ts of source tag 2 are nan
            pass

    sorted_x_array_processed_data = x_array_processed
    sorted_y_array_processed_data = y_array_processed

    #print("load_anom",str(year), " ", counter)
    #print("load_anom_neg", str(year), " ", counter_anom)
    print("anom_out_of_bound ", anom_count)
    print("total valid data ", len(sorted_x_array_processed_data))

    return sorted_x_array_processed_data,sorted_y_array_processed_data

def data_processing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data preprocessing on the input dataframe and transform it into train and validation datasets.

    In this code example, the input dataframe is divided into train_x, train_y, val_x and val_y dataframes.

    Args:
        df: The input dataframe containing the data to be preprocessed.

    Return:
        A dictionary containing the preprocessed data.
    """


    # ADD YOUR CODE HERE
    dataset1_x, dataset1_y = process_core(df)
    dataset1_x_train, dataset1_x_test, dataset1_y_train, dataset1_y_test = train_test_split(dataset1_x,dataset1_y,test_size=0.3, random_state= 1)
    print(dataset1_x)
    print(len(dataset1_x))


    return {"dataset1_x_train": dataset1_x_train,
            "dataset1_y_train": dataset1_y_train,
            "dataset1_x_test": dataset1_x_test,
            "dataset1_y_test": dataset1_y_test,}
