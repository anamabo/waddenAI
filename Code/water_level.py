"""
Module to deal with water level data.
Authors: @fwilms and @marti_cn
"""
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 50)

import utils
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


class WaterLevel(object):
	"""
	Class that deals with water level data.
	"""

	#def __init__(self):
	#def get_raw_data(self)

	@staticmethod
	def inpute_water_level(df,code):
		""" Inputes df and puts everything to freq= 10mins"""
		df = df[df['Meetwaarde.Waarde_Numeriek']  < 900]
		df.reset_index(inplace=True, drop=True)
		df['Tijdstip'] = pd.to_datetime(df['Tijdstip'], format='%Y-%m-%d %H:%M:%S')

		df_wl = df[['Meetwaarde.Waarde_Numeriek','Tijdstip']]
		df_wl.set_index('Tijdstip', inplace=True)
		df_wl = df_wl.loc[~df_wl.index.duplicated(keep='first')]
		df_wl = df_wl.resample('10min').interpolate(method='linear', limit=144) #CAREFUL: UPSAMPLING WITH Forward fill!
		dates = df_wl.index
		start_date = pd.Timestamp(dates.min())
		end_date = pd.Timestamp(dates.max())
		interval = pd.date_range(start_date, end_date, freq='10min')

		df_wl = df_wl.reindex(interval, fill_value=np.nan)
		df_wl = df_wl.interpolate('linear',limit=2)
		df_wl.rename(columns={'Meetwaarde.Waarde_Numeriek': 'water_level_%s'%code}, inplace=True)

		#Reset index and put additional information
		df_wl = df_wl.reset_index().rename(columns={'index': 'date'})
		return df_wl

	@staticmethod
	def resample_wl_data_for_interp(ipath, filename= 'all_postproc_10min.csv'):
		"""From 10 mins to day   """
		#Read water level data
		df = pd.read_csv(os.path.join(ipath,filename))
		df['date'] = pd.to_datetime(df.date)

		#Resample to 1 day
		df = df.set_index('date').resample('D').mean()

		#Remove stations with no data
		a = df.isnull().sum()/df.shape[0]
		remove_these_columns = list(a[a== 1.0].index)
		df.drop(remove_these_columns, axis=1, inplace= True)
		return df

	@staticmethod
	def plotting(path, array, date, ofilename= 'proof.png'):
		fig, ax = plt.subplots(figsize=(8, 8))
		p = ax.imshow(array, cmap= 'Blues', vmin= -150, vmax= 150,)
		ax.set_xlabel('x [pixels]', fontsize = 10)
		ax.set_ylabel('y [pixels]', fontsize = 10)
		ax.set_title(date, fontsize= 12)

		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)

		cbar= fig.colorbar(p, cax=cax)
		cbar.ax.set_ylabel('Water level relative to mean sea level [cm]', fontsize = 10)
		plt.savefig(os.path.join(path, ofilename), bbox_inches= 'tight')

	def preprocessing(self, ipath, opath, all_files= False, files= []):
		"""
		Preprocess the water level data.
		Input:
			ipath: (String). Path where the raw data is located.
			opath: (String). Path where the clean data will be stored.
			all_files: (Boolean). True if all the files in ipath need to be cleaned.
								  False if some files need to be cleaned. Here
								  specify in files the list of files to be cleaned.
								  Default: False.
			files: (list of strings). List of filenames that need to be cleaned.
				   If all_files= False, provide the list.
		Output:
			csv files stored in opath.
			all_postproc_10min.csv: File with all water levels in columns.
		"""
		if (all_files):
			fns = os.listdir(ipath)
		else:
			if (len(files)!=0):
				fns = files
			else:
				raise ValueError('files is an empty list. Provide a list of files for processing.')

		failed_files = []
		for f in fns:
			keyName =  f[:-4]

			print('Reading file {}...'.format(f))
			fn = os.path.join(ipath,f)
			df = pd.read_csv(fn)

			if not df.empty:
				code = df.locatie_code[0]
				cols2keep = ['Tijdstip','Meetwaarde.Waarde_Numeriek']
				df = df[cols2keep]
				df = df[df['Tijdstip'].notnull()]
				df.reset_index(drop= True, inplace= True)

				df_wl = self.inpute_water_level(df, code)
				fn_postproc = os.path.join(opath,'{}_orig_postproc_10min.csv'.format(keyName))
				print('Writing {}_orig_postproc_10min.csv'.format(keyName))
				df_wl.to_csv(fn_postproc, index= False)

			else:
				print('file {} is empty'.format(f))
				failed_files.append(f)

		#Read error file if any....
		if (len(failed_files)!=0):
			fnaam= os.path.join(opath, 'err.txt')
			err_file=open(fnaam,'w')
			err_file.writelines(failed_files)
			err_file.close()

             
    def concat_all_wl_tables(self, ipath, outfilename= 'all_postproc_10min.csv'):
        """
		Inputs:
			ipath: (String). Path where the postprocessed data is stored.
		Output:
		    table in format: date| wl1 |...| wln|
		"""
        DF = []
		for file in os.listdir(ipath):
            if file.endswith(".csv"):
                print('file:', file)
                df= pd.read_csv(os.path.join(ipath,file) )
				df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
				df = df[(df.date>= pd.to_datetime('2016-01-01') ) &
       					(df.date<= pd.to_datetime('2018-12-31') )]
				DF.append(df.set_index('date'))
		DF = pd.concat(DF, sort=True, axis=1)
		DF = DF.reset_index().rename(columns={'index': 'date'})
		DF.to_csv(os.path.join(ipath, outfilename), index= False)


	def prepare_water_level(self, ipath, wlfilename= 'all_postproc_10min.csv'):
		"""
		Input:
			date| wl1 |...| wln|
		Output:
			date| x1|y1|wl1|...|xn|yn|wln|
		"""
		#get the data resampled daily, rm stations with no data
		wlev = self.resample_wl_data_for_interp(ipath, filename= wlfilename)

		#Water level stations are the columns of wlev:
		wl_stations = [i.split('_')[-1] for i in wlev.columns]

		#Get locations of rws stations and select those for which
		# water level data is available
		locs= utils.get_rws_coordinates_from_table()
		locs = locs[locs.index.isin(wl_stations)]
		new_wl_stations= locs.index.values
		print('Number of water level stations:', locs.shape[0])
		print('Stations:', new_wl_stations )

		# Filter wl data with locations
		wlev = wlev[['water_level_%s'%i for i in new_wl_stations]]

		# Obtain a table with this format: date, x1,y1,wl1,..., xn,yn,wln
		interp = pd.DataFrame(index= wlev.index)

		for station in new_wl_stations:
			x, y = locs.loc[station, ['x_28992', 'y_28992']].values
			interp['x_%s'%station] = x
			interp['y_%s'%station] = y
			interp['water_level_%s'%station] = wlev['water_level_%s'%station]

		interp = interp.reset_index()

		# Get columns with missing values:
		missing = wlev.isnull().sum()
		missing =  list(missing[missing != 0].index)

		# For the cols with missing values, put nan in their respective x and y
		for m in missing:
			st = m.split('_')[-1]
			interp.loc[interp[m].isna(), ['x_%s'%st,  'y_%s'%st]] = np.nan
		return interp

	def interpolate(self, interpdf, output_path, generate_plots= True, format= 'csv'):
		"""
		Function that makes Kringe interpolation on the preprocessed wl data.
		Inputs:
		   format: 'csv' : generates csv files in folder output_path/csv_files
				   'tif': generates tif files in folder output_path/tif_files

		"""
		# Get raster of Wadden Sea
		extent, dims, crs, ws = utils.get_raster_wadden_sea()

		# grid to make interpolation on
		x= np.linspace(extent[0], extent[2], dims[0])
		y = np.linspace(extent[1],extent[3], dims[1])

		# For each date, make an interpolation using data, on grid x,y
		#for row in range(interpdf.shape[0]):
		for row in [0]:
			print('Interpolating data at index %d ...'%row)
			date= interpdf.loc[row,'date']
			no_obs = int(len(interpdf.iloc[row, 1:].dropna().values)/3.)
			data = interpdf.iloc[row, 1:].dropna().values.reshape(no_obs, 3)
			# Do the linear interpolation
			UK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
								variogram_model='power', coordinates_type= 'euclidean',
                    			verbose= False, enable_plotting= False)
			# kriged points and the variance.
			z, ss = UK.execute('grid',x,y)
			watlev = z.data*ws
			watlev1 = np.where(watlev !=0, watlev, np.nan)


			if (format == 'tif'):
				#Convert array to raster.
				opath= os.path.join(output_path,'tif_files/')
				if not os.path.exists(opath):
					os.makedirs(opath)
				utils.array2raster(463, extent[0], extent[3], crs, watlev, opath, raster_name= 'wl_%d-%02d-%02d.tif'%(date.year,date.month, date.day) )

			if (format== 'csv'):
				opath= os.path.join(output_path,'csv_files/')
				if not os.path.exists(opath):
					os.makedirs(opath)
				table= utils.arraytocsv(watlev1, dims[0], dims[1])
				table.rename(columns={'value': 'water_level'}, inplace= True)
				table.to_csv(opath+'wl_%d-%02d-%02d.csv'%(date.year,date.month, date.day), index= False)

			#Create plot if true:
			if (generate_plots):
				path= os.path.join(output_path, 'plots/')
				if not os.path.exists(path):
					os.makedirs(path)
				self.plotting(path, watlev1, interpdf.loc[row,'date'] , ofilename= '%05d.png'%row)


def preprocessing_water_level_data():
    """
    Obtain water level, preprocess raw data and make interpolation
    """
    wlpath = os.path.abspath( os.path.join('..', '..', '2_data', 'rws_water_level_and_waves'))
    
    init_path = os.path.abspath( os.path.join(wlpath, 'raw_data'))
    out_path = os.path.abspath( os.path.join(wlpath, 'postprocessed_data','water_level_every_10mins'))
    interp_data_path= os.path.abspath( os.path.join(wlpath,'postprocessed_data', 'interpolated_data'))
    
    raw_files = glob.glob(os.path.join(init_path,'*_OW_WATHTE_NAP_*.csv'))
    raw_files = [i.rsplit('\\', 1)[-1] for i in raw_files]
    print(raw_files[0])
    wl = WaterLevel()
    # preprocessing and concat are tooo long processes. Better to use the data lready in place.
    #wl.preprocessing(init_path, out_path, all_files= False, files= raw_files)
    #wl.concat_all_wl_tables(out_path, outfilename= 'all_postproc_10min.csv')
    #print('end of preprocessing raw water level data.')
    #prep_wl = wl.prepare_water_level(out_path,  wlfilename= 'all_postproc_10min.csv')
    #wl.interpolate(prep_wl, interp_data_path, generate_plots= False, format= 'csv')
    #print('Interpolation of water level data finished.')

if __name__== '__main__':
	preprocessing_water_level_data()
