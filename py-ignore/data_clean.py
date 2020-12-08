import pandas as pd
import numpy as np

def trans_smoker(data):
    if data == 'yes':
        return 1
    elif data == 'no':
        return 0

def data_transform():
    df = pd.read_csv('insurance.csv')
    print(df)
    df['smoker'] = df['smoker'].apply(trans_smoker)
    df = pd.concat([df, pd.get_dummies(df.sex)], axis=1)
    df = pd.concat([df, pd.get_dummies(df.region)], axis=1)
    df = df.drop(columns=['sex', 'region'])
    df.to_csv('trans_insurance.csv', index=False)

def main():
    df = pd.read_csv('trans_insurance.csv')

if __name__=='__main__':
    main()