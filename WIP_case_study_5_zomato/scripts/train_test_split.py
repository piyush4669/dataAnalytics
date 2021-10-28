import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#GET DATASETS

df = pd.read_csv("../data/train_final.csv")
df = df.drop(['id','week','center_id','meal_id'], 1)

#SPLIT DATASET

train, test = train_test_split(df, test_size=0.1)

def one_hot_encoder(dataset):
    city_code = pd.get_dummies(dataset.city_code, prefix='city_code')
    region_code = pd.get_dummies(dataset.region_code, prefix='region_code')
    center_type = pd.get_dummies(dataset.center_type, prefix='center_type')
    category = pd.get_dummies(dataset.category, prefix='category')
    cuisine = pd.get_dummies(dataset.cuisine, prefix='cuisine')
    dataset = dataset.drop(['city_code','region_code','center_type','category','cuisine'],1)
    dataset = pd.concat([dataset,city_code,region_code,center_type,category,cuisine], axis=1)
    return dataset

X = train.drop('num_orders', 1)
X = one_hot_encoder(X)
X.to_csv('X.csv', index=False)
Y = train['num_orders']
Y.to_csv('Y.csv', index=False)


x = test.drop('num_orders', 1)
x = one_hot_encoder(x)
x.to_csv('x.csv', index=False)
y = test['num_orders']
y.to_csv('y.csv', index=False)


