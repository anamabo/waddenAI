# -*- coding: utf-8 -*-
"""
Module that takes the raw data obtained from RWS on salinity,
makes a preprocessing (interpolate on time) and 
interpolates spatially to get an estimate on the whole WS 
@author: marti_cn
"""
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 50)

import utils
from pykrige.ok import OrdinaryKriging

class Salinity(object):
    def __init__(self, rws_data_path):
        """
        Initialization: get paths from root rws_data
        """
        self.ipath=  os.path.abspath( os.path.join(rws_data_path, 'raw_data') )
        
        opath=  os.path.abspath( os.path.join(rws_data_path, 'preprocessed_data', 'salinity_interpolated')  )
        if not os.path.exists(opath):
            os.makedirs(opath)
        self.opath= opath
        self.metadata_path= os.path.abspath( os.path.join(rws_data_path, 'metadata') ) 
        
    @staticmethod
    def plotting(path, array, date, ofilename= 'proof.png'):
        fig, ax = plt.subplots(figsize=(8, 8))
        p = ax.imshow(array, cmap= 'Blues', vmin= 5, vmax= 35,)
        ax.set_xlabel('x [pixels]', fontsize = 10)
        ax.set_ylabel('y [pixels]', fontsize = 10)
        ax.set_title(date, fontsize= 12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar= fig.colorbar(p, cax=cax)
        cbar.ax.set_ylabel('Salinity', fontsize = 10)
        plt.savefig(os.path.join(path, ofilename), bbox_inches= 'tight')
        
    def preprocessing(self):
        """
        Function that cleans TS of raw salinity data and stores them in a list.
        Interpolation in time is carried out.
        The raw_salinity data is obtained with /3_scripts/mudai/get_data_rws_general.py
        """
        #Get all the rws stations 
        locs= utils.get_rws_coordinates_from_table()
        rws_stations = locs.index.values
        
        # Get unique code of salinity
        rws_params = pd.read_excel( os.path.join(self.metadata_path, 'parameters_traslated.xlsx') )
        sc = rws_params[rws_params['Parameter description'] == 'salinity_surface_water']
        salinityCode= sc.iloc[0,3]+'_'+sc.iloc[0,9]+'_'+sc.iloc[0,12]+'_'+sc.iloc[0,15]
        
        #Preprocessing of each file that belongs to one station
        self.salinity = []
        salinity_stations= []
        for j in range(len(rws_stations)):        
            file = rws_stations[j]+'_'+salinityCode+'.csv'
            
            try:
                df = pd.read_csv(os.path.join(self.ipath,file))[['Tijdstip',
                                'Meetwaarde.Waarde_Numeriek', 
                                'Compartiment.code',
                                'Eenheid.code',
                                'Grootheid.code',
                                'Hoedanigheid.code',
                                'Parameter.code']]
                df['rwscode']=  [df.loc[i,'Compartiment.code']+'_'+df.loc[i,'Grootheid.code']+'_'+df.loc[i,'Hoedanigheid.code']+'_'+df.loc[i, 'Parameter.code'] for i in range(df.shape[0])  ]
                df = df[df.rwscode== salinityCode]
                colname= 'Salinity_{}'.format(rws_stations[j]) 
                df.rename(columns={'Tijdstip':'time', 'Meetwaarde.Waarde_Numeriek': colname}, inplace= True)
                df.time = pd.to_datetime(df.time, utc= True).dt.tz_convert(tz= None) # convert to UTC and removes timezone offset
                df.loc[df[colname]>= 10000000, colname] = np.nan 
        
                df = df[(df.time>= pd.to_datetime('2016-01-01') ) & (df.time <= pd.to_datetime('2018-12-31')) ]
                
                if (df.shape[0]> 1):
                    salinity_stations.append(rws_stations[j])
                    df = df.set_index('time').resample('D').sum().reset_index()
                    df.rename(columns= {colname: 'total_'+colname}, inplace= True)
                    df.loc[df['total_'+colname]== 0, 'total_'+colname] = np.nan # This resampling will have 0 data
                    df['total_'+colname].interpolate(method= 'linear', inplace= True)
                    
                    self.salinity.append(df[['time', 'total_'+colname]])
                
            except:
                continue
            # Finally, get the locations of the salinity stations for interpolation
            self.salinity_stations = locs[locs.index.isin(salinity_stations)].reset_index('Code')
            
    def concatenate_clean_tables(self):
        """
        Receives self.salinity obtained in preprocessing()
        and generates a table with this format:
            date| x1|y1|wl1|...|xn|yn|wln|
        NOTE: It's called in interpolate, so in main no need to call this function!
        """
         
        indx = pd.date_range(start= pd.to_datetime('2016-01-01'), end= pd.to_datetime('2018-12-31'), freq= 'D')
        sal = pd.DataFrame({'time': indx})

        for i in range(len(self.salinity)):
            df= self.salinity[i]
            colname = df.columns[1]
            station = colname.split('_')[2]
            df['x_{}'.format(station)] = self.salinity_stations[self.salinity_stations.Code== station]['x_28992'].values[0]
            df['y_{}'.format(station)] =  self.salinity_stations[ self.salinity_stations.Code== station]['y_28992'].values[0]
            df = df[['time', 'x_{}'.format(station), 'y_{}'.format(station), colname]] # reordering of columns
            sal = pd.merge(sal, df, on = 'time', how= 'left')
    
        sal = sal.dropna().reset_index(drop= True)
        # update salinity dataset
        self.salinity = sal

    def interpolate(self, generate_plots= True, format= 'csv'):
        """
        Interpolated the concatenated data.
        """
        self.concatenate_clean_tables() # update salinity to desired structure
        
        # Get of Wadden Sea
        extent, dims, crs, ws = utils.get_raster_wadden_sea()
        # grid to make interpolation on
        x= np.linspace(extent[0], extent[2], dims[0])
        y = np.linspace(extent[1],extent[3], dims[1])
        
        # For each date, make an interpolation using data, on grid x,y
        #for row in [0]:
        for row in range(self.salinity.shape[0]):
            print('Interpolating data at index %d ...'%row)
            date= self.salinity.loc[row,'time']
            no_obs = int(len(self.salinity.iloc[row, 1:].dropna().values)/3.)
            data = self.salinity.iloc[row, 1:].dropna().values.reshape(no_obs, 3) #x, y, salinity

            # Do the linear interpolation
            UK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
								variogram_model='power', coordinates_type= 'euclidean',
                    			verbose= False, enable_plotting= False)
			# kriged points and the variance.
            z, ss = UK.execute('grid',x,y)
            salinity = z.data*ws # on land, salinity is zero
            salinity1 = np.where(salinity !=0, salinity, np.nan)
            
            if (format == 'tif'):
				#Convert array to raster.
                opath= os.path.join(self.opath,'tif_files/')
                if not os.path.exists(opath):
                    os.makedirs(opath)
                utils.array2raster(463, extent[0], extent[3], crs, salinity, opath, raster_name= 'salinity_%d-%02d-%02d.tif'%(date.year,date.month, date.day) )
                
            if (format== 'csv'):
                opath= os.path.join(self.opath,'csv_files/')
                if not os.path.exists(opath):
                    os.makedirs(opath)
                table= utils.arraytocsv(salinity1, dims[0], dims[1])
                table.rename(columns={'value': 'salinity'}, inplace= True)
                table.to_csv(opath+'salinity_%d-%02d-%02d.csv'%(date.year,date.month, date.day), index= False)

            if (generate_plots):
                path= os.path.join(self.opath, 'plots/')
                if not os.path.exists(path):
                    os.makedirs(path)
                self.plotting(path, salinity1, self.salinity.loc[row,'time'], ofilename= '%05d.png'%row)


def preprocessing_salinity_data():
    rwsdata_path = os.path.abspath( os.path.join('..', '..', '2_data', 'rws_data') )
    s= Salinity(rwsdata_path)
    s.preprocessing()  
    s.interpolate(generate_plots= False, format= 'csv')    
    
if __name__== '__main__':
    preprocessing_salinity_data()
       
        
    
