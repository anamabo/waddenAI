# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:05:39 2020

@author: marti_cn
"""

import geopandas as gpd
import pandas as pd
import os

def get_inlets_dataset(ipath, opath, filename= 'inlets.csv', plot_table= True, save_table= True):
         
    inlets = []   
    files = ['Ame_buiten_tot.shp', 'Ame_binnen_tot.shp', 
             'Eems_binnen_tot.shp', 'Eems_buiten_tot.shp',
             'Eld_binnen_tot.shp', 'Eld_buiten_tot.shp',
             'Frz_binnen_tot.shp', 'Frz_buiten_tot.shp',
             'Md_binnen_tot.shp', 'Md_buiten_tot.shp',
             'Vlie_binnen_tot.shp', 'Vlie_buiten_tot.shp']

    names= [f[:-8] for f in files]

    count = 1
    for f in range(len(files)):
        df = gpd.read_file(os.path.join(ipath,files[f]))
        df['inlet'] = names[f]
        df['ID'] = count
        count +=1
        inlets.append(df)
    
    inlets = pd.concat(inlets)

    inlets.reset_index(inplace= True)
    inlets.drop(['index', 'ID'], axis=1, inplace=True)

    if(plot_table):
        inlets.plot()

    # Save as geodf
    inlet = gpd.GeoDataFrame(inlets, 
                             crs= {'init': 'epsg:28992'}, 
                             geometry= inlets.geometry)

    if(save_table):
        inlet.to_csv(os.path.join(opath,filename), index= False)
    return inlet

def get_temporal_features(path, filename='temporal_features.csv', 
                          start_date= '2016-01-01', end_date= '2018-12-31', 
                          frequency= 'D', save_table= True):
    """
    Generates a dataframe with date range from start_date to end_date
    at the given frequency, with columns on season, month and year.
    """
    date= pd.date_range(start= pd.to_datetime(start_date), 
                        end= pd.to_datetime(end_date), freq= frequency)
    time_df= pd.DataFrame(date, columns= ['date'])
    time_df['month'] = time_df.date.dt.month
    time_df['season'] = ((time_df.month%12 + 3)//3)
    time_df['season'] = time_df['season'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'})
    #Dummy-encode the season
    time_df = pd.merge(time_df, 
                       pd.get_dummies(time_df['season']),
                       left_index= True, right_index= True)
    time_df.drop('season', axis=1, inplace= True)
    
    if(save_table):
        time_df.to_csv(os.path.join(path,filename), index= False)
    return time_df


if __name__== '__main__':
    ipath_inlets=  os.path.abspath( os.path.join('..', '..', '2_data', 'Inlets', 'PolygonsVOP' ))
    opath_inlets=  os.path.abspath( os.path.join('..', '..', '2_data','Inlets'))
    inlet_df= get_inlets_dataset(ipath_inlets, opath_inlets, save_table= True)

    ipath_time = os.path.abspath( os.path.join('..','..', '2_data', 'temporal_features' ))
    times = get_temporal_features(ipath_time) 


