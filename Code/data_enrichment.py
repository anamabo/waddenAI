
"""
Module that makes the joins to generate the input for the ML model
Authors: @marti_cn
"""
#%%
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import os

def plot(pivot_table, vmin, vmax, 
         opath='', filename='', savefig= True, title_plot= '',cmap= '',cmap_title= ''):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(8, 8))
    p = ax.imshow(pivot_table.values, cmap= cmap, vmin= vmin, vmax= vmax)
    ax.set_xlabel('x [pixels]', fontsize = 10)
    ax.set_ylabel('y [pixels]', fontsize = 10)
    ax.set_title(title_plot, fontsize= 12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar= fig.colorbar(p, cax=cax)
    cbar.ax.set_ylabel(cmap_title, fontsize = 10)
    if (savefig):
        plt.savefig(os.path.join(opath, filename), bbox_inches= 'tight')
    plt.show()

def load_sediment_dist():
    #Load the mud pct and create the sand pct from it
    # THESE TWO FEATURES ARE UNCORRELATED!
    path = os.path.abspath(os.path.join('..','..', '2_data', 'sediment_data') )
    df = pd.read_csv(os.path.join(path, 'mud_volume_percentage.csv'))  
    df.columns= ['x_meters', 'y_meters', 'pct_mud']
    df['pct_mud'] /= 100.
    df['pct_sand'] = 1-df['pct_mud']
    return df

def load_bathymetry():
    path = os.path.abspath(os.path.join('..','..', '2_data', 'bathimetry_wadden_sea') )
    df = pd.read_csv(os.path.join(path, 'bathymetry_interpolated.csv'))
    df.columns= ['x_meters', 'y_meters', 'bathymetry']
    return df

def load_pixels_to_meters():
    path = os.path.abspath(os.path.join('..','..', '2_data', 'grid') )
    df = pd.read_csv(os.path.join(path, 'wadden_pixel2meter.csv'))
    return df

def load_inlets():
    path = os.path.abspath(os.path.join('..','..', '2_data', 'Inlets') )
    df = pd.read_csv(os.path.join(path, 'inlets.csv'))

    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry = 'geometry', crs= 'epsg:28992')

    return df

def df_to_geodf(df, xcol= 'x_meters', ycol= 'y_meters'):
    geometry = gpd.points_from_xy(df[xcol], df[ycol])
    df = gpd.GeoDataFrame(df, geometry = geometry, crs= 'epsg:28992')
    return df

def get_TS_data(date, path, prefix= 'spm'):
    # date must me timestamp
    try:
        year, month, day = date.year, date.month, date.day
        df = pd.read_csv(os.path.join(path, '%s_%d-%02d-%02d.csv' % (prefix,year, month, day)))
        df['date'] = date
        if'pixelID' not in df.columns:
            df['pixelID'] = np.arange(df.shape[0])
        if prefix == 'mesh2d_hwav': # this is wave data
            df.rename(columns={'Value': 'H0'}, inplace=True)
            prefix= 'H0'

        df = df[['pixelID', 'date', prefix]]
        return df
    except:
        #Here the file does not exist; i.e. no variable for this particular date
        if prefix == 'mesh2d_hwav':  # this is wave data
            prefix = 'H0'
        df= pd.DataFrame({'pixelID': np.arange(76032),
                          'date': date,
                          prefix: np.nan})
        return df

def get_spm(date):
    # date must me timestamp
    spm_path = os.path.abspath(os.path.join('..', '..', '2_data', 'Remote_sensing', 'spm'))

    try:
        year, month, day = date.year, date.month, date.day
        spm = pd.read_csv(os.path.join(spm_path, 'spm_%d-%02d-%02d.csv' % (year, month, day)))
        spm.drop(['xpixel', 'ypixel', 'xcoord', 'ycoord'], axis=1, inplace= True)
        spm['date'] = date
        spm = spm[['pixelID', 'date', 'spm']]
        return spm
    except:
        # Here the file does not exist; i.e. no spm for this particular date
        spm= pd.DataFrame({'pixelID': np.arange(76032),
                          'date': date,
                          'spm': np.nan})
        return spm

#%%
##main
#########################################################
########joins of variables time indep ###################
#########################################################
sediment = load_sediment_dist()
bathy = load_bathymetry()
pixels= load_pixels_to_meters()
inlets= load_inlets()

print('sediment shape:', sediment.shape)
print('bathy shape:', bathy.shape)
print('pixels shape:', pixels.shape)


#join sediment and bathy. Join by index since x,y are not pixels
data = pd.merge(sediment, bathy[['bathymetry']], left_index= True, right_index= True)
print('sed+ bathy:', data.shape)

#join with pixels
data = pd.merge(pixels, data[['pct_mud', 'pct_sand', 'bathymetry']], left_index= True, right_index= True )
print('sed+bathy+pixel:', data.shape)

#Filter points that are not in the wadden sea
data = data[data.value==1]
print('shape only wadden:', data.shape)

#Still there are some missing values in sediment and bathy, remove them
data= data.dropna()
print('shape rm missing in mud:', data.shape)

data = df_to_geodf(data)

##plot sediment data to check correctness
#s = pd.pivot_table(data, index= 'y_pixel', columns= 'x_pixel', values= 'pct_mud', dropna= False)
#opath = os.path.abspath(os.path.join('..','..', '4_plots', 'mud_percentage') )
#plot(s, 0, 1, opath, 'mud.png', cmap = 'YlOrBr' , cmap_title= 'Mud percentage')


#spatial join with inlets
data = gpd.sjoin(data, inlets, how='inner', op='intersects')
data.drop(['geometry', 'index_right'], axis=1, inplace=True)
print('shape after spatial join:', data.shape)

# Write dataset.
# the pixelsID of this dataset are those used for modelling
# Joins with TS will be using those points
data.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'spatial_features.csv')), index= False )

del data
del sediment
del bathy
del inlets
del pixels
#%%
######################################################
########joins of time series #########################
######################################################
date_range= pd.date_range(start= pd.to_datetime('2016-04-26'), end= pd.to_datetime('2018-12-31'), freq= 'D')
ids= pd.read_csv(os.path.abspath(os.path.join('..','..', '2_data', 'grid', 'filtered_pixels.csv')) )

date_df = []
# for one date, read each variable and crete a table
for i in range(len(date_range)):
    print(date_range[i])
    spm= get_spm(date_range[i])
    spm= pd.merge(ids, spm, on= 'pixelID', how= 'inner')

    salinity= get_TS_data(date_range[i],
                          os.path.abspath(os.path.join('..', '..', '2_data','rws_data','preprocessed_data','salinity_interpolated', 'csv_files')),
                          prefix= 'salinity')
    salinity= pd.merge(ids, salinity, on= 'pixelID', how= 'inner')

    surge = get_TS_data(date_range[i],
                          os.path.abspath(os.path.join('..', '..', '2_data','surge','interpolated', 'csv_files')),
                          prefix= 'surge')
    surge= pd.merge(ids, surge, on= 'pixelID', how= 'inner')

    chl1 = get_TS_data(date_range[i],
                          os.path.abspath(os.path.join('..', '..', '2_data','Remote_sensing','chlorophyll')),
                          prefix= 'chl_NN')
    chl1= pd.merge(ids, chl1, on= 'pixelID', how= 'inner')

    chl2 = get_TS_data(date_range[i],
                          os.path.abspath(os.path.join('..', '..', '2_data','Remote_sensing','chlorophyll')),
                          prefix= 'chl_OC4ME')
    chl2= pd.merge(ids, chl2, on= 'pixelID', how= 'inner')

    h0 = get_TS_data(date_range[i],
                            'P:/11203755-019-mudai/2_data/gridded_mesh2dhwav',
                            prefix= 'mesh2d_hwav')
    h0= pd.merge(ids, h0, on= 'pixelID', how= 'inner')

    # join all
    data = pd.merge(salinity, surge, on= ['pixelID', 'date'], how= 'inner')
    data = pd.merge(data, h0, on= ['pixelID', 'date'], how= 'inner')
    data = pd.merge(data, spm, on= ['pixelID', 'date'], how= 'inner')
    data = pd.merge(data, chl1, on= ['pixelID', 'date'], how= 'inner')
    data = pd.merge(data, chl2, on= ['pixelID', 'date'], how= 'inner')
    date_df.append(data)
    # free up memory
    del spm
    del salinity
    del surge
    del chl1
    del chl2
    del h0
    del data

date_df = pd.concat(date_df, ignore_index=True)
# join the table with the other time features; i.e. seasons
timef=  pd.read_csv(os.path.abspath(os.path.join('..','..', '2_data', 'temporal_features', 'temporal_features.csv')) )
timef['date'] = pd.to_datetime(timef['date'])

date_df = pd.merge(date_df, timef, on='date', how= 'inner')
date_df.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'temporal_features.csv.gz')),
               index= False, compression= 'gzip')

del ids
del date_range
del date_df
del timef
#%%
######################################################
########join space and time features #################
######################################################
# Read the spatial features
space= pd.read_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'spatial_features.csv')) )
#remove duplicated pixelIDs: let's assume one pixel belongs to an unique inlet
space.drop_duplicates(subset= ['pixelID'], inplace= True)

# read the temporal features
time = pd.read_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'temporal_features.csv.gz')))
time['date'] = pd.to_datetime(time['date'])

#To see pct of missing values in best spm data: 38%
#a = time[time['date']== pd.to_datetime('2018-05-14')]
#print(a.spm.isnull().sum()/float(a.shape[0]))

# join both tables
data = pd.merge(time, space, on= 'pixelID', how= 'inner')
data = df_to_geodf(data, xcol= 'x_meters', ycol= 'y_meters') # this process takes several minutes. Be patient!

#free up memory
del time
del space

# read wind data
wind= pd.read_csv(os.path.abspath(os.path.abspath( os.path.join('..',
                                                                '..',
                                                                '2_data',
                                                                'knmi_meteo', 'preprocessed',
                                                                'integrated_wind_speed_and_direction_projected.csv')) ) )

wind['date'] = pd.to_datetime(wind['date'])
wind['geometry'] = wind['geometry'].apply(wkt.loads)
wind = gpd.GeoDataFrame(wind, geometry = 'geometry', crs= 'epsg:28992')

# last, make the join with wind
date_range= pd.date_range(start= pd.to_datetime('2016-04-26'), end= pd.to_datetime('2018-12-31'), freq= 'D')

final_data = []

for i in range(len(date_range)):
    w = wind[wind['date'] == date_range[i]]
    d = data[data['date'] == date_range[i]]
    df = gpd.sjoin(d, w, how="inner", op='intersects')
    df.drop(['geometry', 'index_right', 'date_right', 'name'], axis=1, inplace=True)
    df.rename(columns={'date_left': 'date'}, inplace=True)
    final_data.append(df)
    print(date_range[i], df.shape)

final_data = pd.concat(final_data, ignore_index= True)
final_data.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'joined_dataset.csv.gz')) )

# Create final tables
#Model1: 2017 only, waves as predictor
#Model2: 2017 only, without waves.
#Model3: 2016-2017, without waves
#Model4: 2016-2017, without chlorophyll -> to get spm when chl info is missing

spm = final_data[final_data['spm'].notnull()]
nospm = final_data[final_data['spm'].isnull()] # For further evaluation in phaseIII
nospm.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'used_for_evaluation.csv.gz')) )

model1 = spm[spm.H0.notnull()]
model1 = model1.reset_index().drop('index', axis=1)
model1 = pd.merge(model1, pd.get_dummies(model1.inlet), left_index= True, right_index= True  )
model1.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'model1.csv.gz')) )

model2= model1.drop('H0', axis=1)
model2 = model2.reset_index().drop('index', axis=1)
model2 = pd.merge(model2, pd.get_dummies(model2.inlet), left_index= True, right_index= True  )
model2.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'model2.csv.gz')) )

model3 = spm.drop('H0', axis=1)
model3 = model3[model3.salinity.notnull()]
model3 = model3.reset_index().drop('index', axis=1)
model3 = pd.merge(model3, pd.get_dummies(model3.inlet), left_index= True, right_index= True  )
model3.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'model3.csv.gz')) )

model4 = spm.drop(['H0', 'chl_NN', 'chl_OC4ME'], axis=1)
model4= model4[model4.salinity.notnull()]
model4= model4.reset_index().drop('index', axis=1)
model4 = pd.merge(model4, pd.get_dummies(model4.inlet), left_index= True, right_index= True  )
model4.to_csv(os.path.abspath(os.path.join('..','..', '2_data', 'final_set_waddenai', 'model4.csv.gz')) )




