"""
Script that creates train and test sets.
The code data_enrichment should be ran first.

@author: marti_cn
"""
import pandas as pd
from utils import split_dataset
import os

def obtain_train_test(path, ifilename, ftrain_name, ftest_name, frac_test= 0.2):

    df = pd.read_csv(os.path.join(path, ifilename))
    df['date'] = pd.to_datetime(df['date'])

    if 'Unnamed: 0'in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace= True)
        
    groups = list(df['inlet'].unique())
    dftrain, dftest = split_dataset(df, frac_test=frac_test, groups=groups) #80/20% for all inlets

    if ((dftrain.shape[0] + dftest.shape[0])!= df.shape[0]):
        ValueError('The shapes of the resulting files are inconsistent with the shape of input table!')
    else:
        dftrain.to_csv(os.path.join(path, ftrain_name), index= False)
        dftest.to_csv(os.path.join(path, ftest_name), index= False)
        return dftrain, dftest

if __name__=='__main__':
    """     
    obtain_train_test(os.path.abspath(os.path.join('..', '..', '2_data', 'enriched_data')),
                      'model3.csv.gz',
                      'model3_train.csv.gz',
                      'model3_test.csv.gz',
                      frac_test=0.2)

    obtain_train_test(os.path.abspath(os.path.join('..', '..', '2_data', 'enriched_data')),
                      'model4.csv.gz',
                      'model4_train.csv.gz',
                      'model4_test.csv.gz',
                      frac_test=0.2)
    """
    train= pd.read_csv('../../2_data/enriched_data/model3_train.csv.gz')
    train = train.sample(n= 1500000, random_state=12345)
    train['date'] = pd.to_datetime(train['date'])

    train.to_csv('../../2_data/enriched_data/model3_trainsampled.csv.gz')