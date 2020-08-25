# -*- coding: utf-8 -*-
"""
Module that takes time series of postprocessed meteorological
data and gets integrated wind speed and direction of max wind.
@author: marti_cn
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt
import utils
import os


class WindData(object):
    def __init__(self, path, wind_filename, voronoi_filename, crs_voronoi='epsg:28992'):
        """
        Uses postprocessed data to:
            1. Put voronoids into Amesfort system
            2. Get wind max, wind avg and wind dir at max (Integrated wind)
            ## METEOROLOGICAL DATA
            ## preprocessing made with script: meteorological_data(get_clean_analyze).ipynb in 3_scripts/mudai/
            ## Voronoids made with script: Voronoids_in_meteodata.ipynb  in 3_scripts/mudai/
            """
        self.path = path

        self.meteo = pd.read_csv(os.path.join(self.path, wind_filename),
                                 index_col='date',
                                 dayfirst=True,
                                 parse_dates=True,
                                 infer_datetime_format=True
                                 )

        self.meteo = self.meteo.reset_index('date')  # from this table we get the wind

        self.voronoids = pd.read_csv(os.path.join(path, voronoi_filename))
        self.voronoids = gpd.GeoDataFrame(self.voronoids, crs={'init': crs_voronoi},
                                          geometry=self.voronoids['geometry'].apply(wkt.loads))
        self.wind = None

    def filter_wind_in_time_range(self, start_date='2016-01-01', end_date='2018-12-31'):
        # Filter  wind data and date from postptocessed meteo data
        cols = ['date', 'name', 'daily_avg_wind_speed', 'wind_direction_grades']
        self.wind = self.meteo[cols]
        self.wind = self.wind[(self.wind.date >= pd.to_datetime(start_date)) &
                              (self.wind.date <= pd.to_datetime(end_date))]

        # Join with voroids data
        self.wind = pd.merge(self.wind, self.voronoids, on='name')


    @staticmethod
    def get_angle_of_max_velocity(lagged_vel_df, lagged_angle_df, anglename, laggval):
        """
        Get angle for which lagged_vel_df is maximum.
        INPUT TABLES CAN NOT CONTAIN DUPLICATED VALUES!
        """
        # For a date, get the column of max value of velocity
        maxidx = lagged_vel_df.idxmax(axis=1)
        # change column name for that of the angle. This has a series of
        # column names (per index= time) of the angle where vel is max.
        maxidx = maxidx.apply(lambda x: anglename+x[-5:])

        # get angle of max velocity
        angle_of_max = []
        for i in range(lagged_vel_df.shape[0]):
            index = maxidx.index[i]
            column = maxidx[i]
            angle_of_max.append(lagged_angle_df.loc[index, column])

        angle_of_max = pd.DataFrame(angle_of_max, index=lagged_angle_df.index,
                                    columns=['dir_max_wind_vel_%d_days_ago'%laggval])

        return angle_of_max

    def get_integrated_wind_speed_and_angle(self, save_table=True,
                                            output_filename='integrated_wind_speed_and_direction_projected.csv'):
        """
        Function to obtain max Vel, dir of max Vel and avg Vel
        """
        self.wind.set_index('date', inplace=True)

        locations= self.wind.name.unique()
        int_wind_speed = []

        for i in range(len(locations)):
            #print(locations[i])
            df = self.wind[self.wind.name==locations[i]][['daily_avg_wind_speed']].reset_index().drop_duplicates(subset='date').set_index('date')
            df1 = self.wind[self.wind.name==locations[i]][['wind_direction_grades']].reset_index().drop_duplicates(subset='date').set_index('date')

            for lagged_terms in range(1, 9):
                #print(lagged_terms)
                lag_wind = utils.convert_ts_to_ml_problem(df, lagged_terms=lagged_terms, forecast_terms=0)
                lag_angle= utils.convert_ts_to_ml_problem(df1, lagged_terms=lagged_terms, forecast_terms=0)

                # Get maximun wind speed, angle of max speed and avg wind speed
                theta= self.get_angle_of_max_velocity(lag_wind, lag_angle,'wind_direction_grades', lagged_terms)
                res= pd.merge(lag_wind.max(axis=1).reset_index(), lag_wind.mean(axis=1).reset_index() , on='date')
                res.columns= ['date', 'max_wind_speed_%d_days_ago'%lagged_terms, 'avg_wind_speed_%d_days_ago'%lagged_terms]
                res= pd.merge(res, theta.reset_index(), on='date')

                if lagged_terms == 1:
                    wind_vel_per_loc = res.copy()
                else:
                    wind_vel_per_loc = pd.merge(wind_vel_per_loc, res, on='date', how='inner')

            wind_vel_per_loc['name']= locations[i]
            int_wind_speed.append(wind_vel_per_loc)

        int_wind_speed = pd.concat(int_wind_speed, ignore_index=True)

        #Merge with meteo data to get the geometry of the voronoids
        #geoms = self.meteo[['name', 'geom']].groupby('name').first().reset_index()

        int_wind_speed= pd.merge(int_wind_speed, self.voronoids, on=['name'], how='inner' )

        if save_table:
            int_wind_speed.to_csv(os.path.join(self.path,output_filename), index=False)
            return None
        else:
            return int_wind_speed


def preprocessing_wind_data():
    ## METEOROLOGICAL DATA
    ## preprocessing made with script: meteorological_data(get_clean_analyze).ipynb
    ## Voronoids made with script: Voronoids_in_meteodata.ipynb
    ## Here voronoids are converted into Amesfort system
    ## Here the proper calculation of wind speed and angle at max vel is done.
    
    path =  os.path.abspath( os.path.join('..', '..', '2_data', 'knmi_meteo', 'preprocessed'))
    windfilename = 'meteorological_data_withgeom.csv'
    voroifilename = 'meteo_voroids_amesfort.csv' #'meteo_voroids_epsg4326.csv'

    # Get wind data filtered in timerange of project and in wgs84 system
    w = WindData(path, windfilename, voroifilename, crs_voronoi='epsg:28992') #'epsg:4326'
    w.filter_wind_in_time_range()
    # Get Integrated wind speed
    w.get_integrated_wind_speed_and_angle(save_table=True)
    return None


if __name__ == '__main__':
    preprocessing_wind_data()


