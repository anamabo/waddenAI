"""
Script that reads the raster files of spm
and chrolophyll to transform them into tables.
"""

import os
from utils import Preprocessing_raster
import pandas as pd
import numpy as np

def get_spm_data(ipath, opath):
    files = os.listdir(ipath)
    for i in range(len(files)):
        date = pd.to_datetime(files[i].split('T')[0])
        print(i, date)
        p = Preprocessing_raster(os.path.join(ipath, files[i]))
        p.get_table_from_raster(colval='spm', transform_sys=False, pixel_id=True)
        df = p.table.copy()
        # Replace -9999 by nan
        df = df.replace(-9999, np.nan)
        # save file
        filename = 'spm_%d-%02d-%02d.csv' % (date.year, date.month, date.day)
        df.to_csv(os.path.join(opath, filename), index=False)

    return None


def get_chlorophyll_data(ipath, opath, prefix = 'OC4ME'):
    files = os.listdir(ipath)
    for i in range(len(files)):
        date = pd.to_datetime(files[i].split('T')[0])
        print(i, date)
        p = Preprocessing_raster(os.path.join(ipath, files[i]))
        p.get_table_from_raster(colval='chl_%s'%prefix, transform_sys=False, pixel_id=True)
        df = p.table.copy()
        # Replace -9999 by nan
        df = df.replace(-9999, np.nan)
        # save file
        filename = 'chl_%s_%d-%02d-%02d.csv' % (prefix, date.year, date.month, date.day)
        df.to_csv(os.path.join(opath, filename), index=False)
    return None

if __name__== '__main__':
    #Run spm
    spm_ipath = 'P:/11203755-019-mudai/2_data/Remote_sensing/Sentinel/TSM_NN/final/'
    spm_opath = os.path.abspath(os.path.join('..', '..', '2_data', 'Remote_sensing', 'spm'))
    get_spm_data(spm_ipath, spm_opath)

    #Run chl obtained with ANNs
    cl_ipath = 'P:/11203755-019-mudai/2_data/Remote_sensing/Sentinel/CHL_NN/final'
    cl_opath = os.path.abspath(os.path.join('..', '..', '2_data', 'Remote_sensing', 'chlorophyll'))
    get_chlorophyll_data(cl_ipath, cl_opath, prefix='NN')

    # Run chl obtained using OC4ME
    cl_ipath = 'P:/11203755-019-mudai/2_data/Remote_sensing/Sentinel/CHL_OC4ME/final'
    cl_opath = os.path.abspath(os.path.join('..', '..', '2_data', 'Remote_sensing', 'chlorophyll'))
    get_chlorophyll_data(cl_ipath, cl_opath, prefix='OC4ME')


