import pandas as pd
import numpy as np

#GET DATASETS

# sample_sub = pd.read_csv("../data/sample_submission.csv")
fulfilment_center_info = pd.read_csv("../data/fulfilment_center_info.csv")
meal_info = pd.read_csv("../data/meal_info.csv")
test_data = pd.read_csv("../data/test.csv")
train_data = pd.read_csv("../data/train.csv")

#CHECK IF MISSGING VALUES IN DATASETS

# print(test_data.isnull().values.any())

#GENERATE FINAL DATASETS BY MERGING

    #TRAIN

train_final_data = pd.merge(left=train_data, right=fulfilment_center_info, how='left', left_on='center_id', right_on='center_id')

train_final_data = pd.merge(left=train_final_data, right=meal_info, how='left', left_on='meal_id', right_on='meal_id')

train_final_data.to_csv('train_final.csv', index=False)

    #TEST

test_final_data = pd.merge(left=test_data, right=fulfilment_center_info, how='left', left_on='center_id', right_on='center_id')

test_final_data = pd.merge(left=test_final_data, right=meal_info, how='left', left_on='meal_id', right_on='meal_id')

test_final_data.to_csv('test_final.csv', index=False)


