"""
Module that takes timeseries of surge data (outcome of matlab scripts)
and makes spatial interpolation to estimate surge on the whole WZ.
Authors: @fwilms and @marti_cn
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


class Surge(object):
    """
	Class that interpolates the TS of Surge. The scripts to obtain
    surge from water level data are written in Matlab.
	"""

    def __init__(self, ipath, opath):
        """
        Initialization Class:
            ipath: Location of surge files made by Bob
            opath: Location of the interpolated files
        """
        self.ipath= ipath
        if not os.path.exists(opath):
            os.makedirs(opath)
        self.opath= opath

    @staticmethod
    def plotting(path, array, date, ofilename= 'proof.png'):
        fig, ax = plt.subplots(figsize=(8, 8))
        p = ax.imshow(array, cmap= 'Blues', vmin= -1.5, vmax= 2,)
        ax.set_xlabel('x [pixels]', fontsize = 10)
        ax.set_ylabel('y [pixels]', fontsize = 10)
        ax.set_title(date, fontsize= 12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar= fig.colorbar(p, cax=cax)
        cbar.ax.set_ylabel('Surge [m]', fontsize = 10)
        plt.savefig(os.path.join(path, ofilename), bbox_inches= 'tight')

    def concat_surge_tables(self):
        """
        Makes the table that will be used in the interpolation
            Input: timeseries of the surge per station
            Output:
                table in format: date| x1|y1|wl1|...|xn|yn|wln|
        """
        DF = []
        for file in os.listdir(self.ipath):
            station= file.split('_')[2]

            if file.endswith(".csv"):
                df= pd.read_csv(os.path.join(self.ipath,file), 
                                header= None, names= ['date', 'surge_%s'%station] )
                df['date'] = pd.to_datetime(df['date'] )
                df = df[(df.date>= pd.to_datetime('2016-01-01') ) & (df.date<= pd.to_datetime('2018-12-31') )]
                DF.append(df.set_index('date'))

        DF = pd.concat(DF, sort=True, axis=1)
        DF = DF.reset_index().rename(columns={'index': 'date'})

        #Resample to 1 day
        DF = DF.set_index('date').resample('D').mean()
        a = DF.isnull().sum()/DF.shape[0]
        remove_these_columns = list(a[a== 1.0].index)
        DF.drop(remove_these_columns, axis=1, inplace= True)
        
        cpath= os.path.abspath(os.path.join('..', '..', '2_data', 'surge', 'concatenated'))
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        DF.to_csv(os.path.join(cpath,'surge_wadden.csv'))

        stations= [i.split('_')[-1] for i in DF.columns]

        #Get locations of stations
        locs= utils.get_rws_coordinates_from_table()
        locs = locs[locs.index.isin(stations)]

        # Obtain a table with this format: date, x1,y1,wl1,..., xn,yn,wln
        interp = pd.DataFrame(index= DF.index)
        for station in stations:
            x, y = locs.loc[station, ['x_28992', 'y_28992']].values
            interp['x_%s'%station] = x
            interp['y_%s'%station] = y
            interp['surge_%s'%station] = DF['surge_%s'%station]

        interp = interp.reset_index()

        # Get columns with missing values:
        missing = DF.isnull().sum()
        missing =  list(missing[missing != 0].index)

		# For the cols with missing values, put nan in their respective x and y
        for m in missing:
            st = m.split('_')[-1]
            interp.loc[interp[m].isna(), ['x_%s'%st, 'y_%s'%st]] = np.nan

        return interp

    def interpolate(self, generate_plots= True, format= 'csv'):
        """
		Function that makes Kringe interpolation on the concatenated surge data.
		Inputs:
		   format: 'csv' : generates csv files in folder output_path/csv_files
				   'tif': generates tif files in folder output_path/tif_files

		"""
        # Get the surge in format: date| x1|y1|wl1|...|xn|yn|wln|
        surgedf= self.concat_surge_tables()

        # Get raster of Wadden Sea
        extent, dims, crs, ws = utils.get_raster_wadden_sea()

		# grid to make interpolation on
        x= np.linspace(extent[0], extent[2], dims[0])
        y = np.linspace(extent[1],extent[3], dims[1])

        # For each date, make an interpolation using data, on grid x,y
        #for index, row in surgedf.iterrows():
        #    date= row['date']
        #for row in [0]:
        for row in range(surgedf.shape[0]):
            print('Interpolating data at index %d ...'%row)
            date= surgedf.loc[row,'date']
            no_obs = int(len(surgedf.iloc[row, 1:].dropna().values)/3.)
            data = surgedf.iloc[row, 1:].dropna().values.reshape(no_obs, 3)

            # Do the linear interpolation
            UK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
								variogram_model='power', coordinates_type= 'euclidean',
                    			verbose= False, enable_plotting= False)
			# kriged points and the variance.
            z, ss = UK.execute('grid',x,y)
            surge = z.data*ws # To get the surge in points inside Wadden and not in the whole bbox
            surge1 = np.where(surge !=0, surge, np.nan)

            if (format == 'tif'):
				#Convert array to raster.
                opath= os.path.join(self.opath,'tif_files/')
                if not os.path.exists(opath):
                    os.makedirs(opath)
                utils.array2raster(463, extent[0], extent[3], crs, surge, opath, raster_name= 'surge_%d-%02d-%02d.tif'%(date.year,date.month, date.day) )

            if (format== 'csv'):
                opath= os.path.join(self.opath,'csv_files/')
                if not os.path.exists(opath):
                    os.makedirs(opath)
                table= utils.arraytocsv(surge1, dims[0], dims[1])
                table.rename(columns={'value': 'surge'}, inplace= True)
                table.to_csv(opath+'surge_%d-%02d-%02d.csv'%(date.year,date.month, date.day), index= False)

            if (generate_plots):
                path= os.path.join(self.opath, 'plots/')
                if not os.path.exists(path):
                    os.makedirs(path)
                self.plotting(path, surge1, surgedf.loc[row,'date'] , ofilename= '%05d.png'%row)


def preprocessing_surge_data():
    """
	Obtain surge from Bob output and interpolate on the Wadden Sea
	"""
    init_path = os.path.abspath( os.path.join('..', '..', '2_data', 'surge','raw','csv_2016-2018' ) )
    out_path =  os.path.abspath( os.path.join('..', '..','2_data', 'surge', 'interpolated' ))

    surge= Surge(init_path, out_path)
    surge.interpolate(format='csv', generate_plots= False)


if __name__== '__main__':
    preprocessing_surge_data()
    
