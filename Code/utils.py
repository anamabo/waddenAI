"""
Script containing generic functions for WaddenAi
@author: marti_cn
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
import geopandas as  gpd
from rasterio.features import shapes
import os
import sklearn.metrics

def convert_coordinate_system(x, y, insys= 'epsg:4326', outsys= 'epsg:28992'):
    """
    Function that converts between spatial coordinate systems
    :x: float or list of floats with the x coordinates to be converted
    :y: float or list of floats with the y coordinates to be converted
    :crs_in (string): coordinate system where x, y are defined
    :crs_out (string): coordinate system of the transformation
    returns
    :x, y: float or lists with the transformation
    """
    transformer = Transformer.from_crs(insys, outsys)
    xt, yt = transformer.transform(x, y)
    return xt, yt

def get_rws_coordinates_from_table():
    """
    Returns a table all the WS stations with their locations, no duplicates.
    """
    path= os.path.abspath( os.path.join('..', '..', '2_data', 'rws_data', 'metadata' ) )
    ws= pd.read_excel(os.path.join(path,'all_locations.xlsx'))
    ws.drop_duplicates(subset= ['Code'], inplace= True)
    ws= ws.set_index('Code')
    return ws

def get_raster_wadden_sea():
    """
    Obtain the spatial information from the Wadden Sea raster.
    Outputs: extent, dimension, coordinatesys, band
    """
    path= os.path.abspath( os.path.join('..', '..', '2_data', 'grid' ) )

    raster = rasterio.open( os.path.join(path,'wadden.tif') )
    xmin, ymin, xmax, ymax = raster.bounds[0], raster.bounds[1], raster.bounds[2], raster.bounds[3]
    widthr, heightr = raster.width, raster.height
    crs = raster.crs
    array = raster.read(1)
    return (xmin, ymin, xmax, ymax), (widthr, heightr), crs, array

def array2raster(pixel_size, xmin, ymax, crs, array, output_path, raster_name= 'raster.tif'):
    """
    Function that exports a numpy array into a raster.
    Inputs:
        pixel_size: (int). Pixel Size. X,Y dimensions must be the same.
        xmin, ymax: (float). Coordinates of the bounding box of the output raster.
        crs: (string). coordinate system of the raster.
        output_path: (string). path to store raster file.
        raster_name: (string). Name of the raster.
    Output:
     None. The raster will be stored in the path specified.
    """
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    new_dataset = rasterio.open(output_path+raster_name, 'w',
                                driver='GTiff',
                                height = array.shape[0],
                                width = array.shape[1],
                                count=1, dtype=str(array.dtype),
                                crs= crs,
                                transform=transform)

    new_dataset.write(array, 1)
    new_dataset.close()

def raster2vector(ipath, rasterName= '', shapeName= ''):
    """
    Convert a raster to vector. In the process, a geodata frame is generated.
    Funcion based on this link: https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons
    Inputs:
        ipath: (String). Path of the raster to be converted
        rasterName: (String). Name of the raster file.
        shapeName: (String). Name of the output shape file.
                    It will be saved in ipath.
    """
    mask = None
    with rasterio.open(os.path.join(ipath,raster)) as src:
        image = src.read(1) # first band, numpy array
        image = np.where(image !=0, image, np.nan) #!OPTIONAL: zero values to nan
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
        in enumerate(shapes(image, mask=mask, transform=src.transform )))

    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    #gpd_polygonized_raster.plot()
    gpd_polygonized_raster.to_file(os.path.join(ipath,shapeName) )

#Deprecated function. Better to use class below
def arraytocsv(array, xmax, ymax):
    """
    Converts an array to dataframe.
    Inputs:
        array: 2D numpy array.
        xmax: (float). width of the array.
        ymax: (float). height of the array.
    Output:
        dataframe with column names: x_pixel, y_pixel, value
    """
    values= []
    X= []
    Y = []
    for x in range(0, xmax):
        for y in range(0, ymax):
            X.append(x)
            Y.append(y)
            values.append(array[y,x])

    df = pd.DataFrame({'x_pixel': X, 'y_pixel': Y, 'value': values})
    return df


class Preprocessing_raster(object):
    """
    Class that reads a raster and converts it into a table with coordinates.
    Usage:
    p = Preprocessing_raster(ipath= '../Data/mekong/building_density/', rasterfilename= 'resampled_building_density.tif')
    table= p.get_table(colval= 'building_info')
    """
    def __init__(self,  rasterfilename=''):
        self.filename = rasterfilename
        self.raster = None
        self.table = None
        self.array = None

    def read_raster(self):
        self.raster = rasterio.open(self.filename)
        self.array = self.raster.read(1)

    @staticmethod
    def from_array_to_table(array, colval='value'):
        """
        Function that converts an array into a table
        Output: table with columns: y, x, value
        """
        h, w = array.shape[0], array.shape[1]
        table = pd.DataFrame(array, index=np.arange(h), columns=np.arange(w))
        table = table.reset_index()
        table.rename(columns={'index': 'y'}, inplace=True)
        table = pd.melt(table, id_vars=['y'], value_vars=np.arange(w))
        table.columns = ['ypixel', 'xpixel', colval]
        return table

    def get_table_from_raster(self, colval='value', transform_sys=True, crs="EPSG:32648", pixel_id= False):
        self.read_raster()
        self.table= self.from_array_to_table(self.array, colval=colval)
        # Add pixel id:
        if pixel_id:
            self.table['pixelID'] = np.arange(self.table.shape[0])
            self.table= self.table[['pixelID', 'xpixel', 'ypixel', colval]]
            cols= self.table.columns

        else:
            self.table = self.table[['xpixel', 'ypixel', colval]]
            cols= self.table.columns

        # Add coordinates per pixel
        xm, ym = self.raster.transform * (self.table['xpixel'].values, self.table['ypixel'].values)
        self.table['xcoord'] = xm
        self.table['ycoord'] = ym
        # if convert = True, convert to desired csr
        if (transform_sys):
            xt, yt = convert_coordinate_system(self.table['xcoord'].values, self.table['ycoord'].values,
                                               crs_in=self.raster.crs,
                                               crs_out=crs)
            self.table['xtrans'] = xt
            self.table['ytrans'] = yt
            order_cols = list(cols[0:-1]) +['xcoord', 'ycoord', 'xtrans', 'ytrans', colval]
            self.table = self.table[order_cols]
        else:
            order_cols = list(cols[0:-1]) + ['xcoord', 'ycoord', colval]
            self.table = self.table[order_cols]
        return None




def convert_ts_to_ml_problem(dataset, lagged_terms= 0, forecast_terms= 2):
    '''
    Function to convert a dataframe of time series into a machine learning problem.
    This function should be used in case you want to use ensemble of trees or  NN for TS forecasting
    For more information on this function, you can go to the following links:
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    LET OP: This function removes missing values; therefore, the shape of the output table
            MIGHT BE different from the input.

            You might want to remove some columns of the output table, depending of the target.

    The output of this function can be used in ensemble of trees or NN.
    Inputs: dataset -> pandas dataframe. The index must be the time.
            lagged_terms -> the time units in the past used to build a TS model.
                            Each lagged term is a column in the output table.
            forecast_terms-> the time units in the future used to make a forecast.
                            Each forecast term is a column in the table.
    Outputs: df -> A dataframe without missing values ready to be used for modelling.
    '''

    cols, names = list(), list()

    # To get lagged terms (t-1, t-2,..., t-n  used for modelling)
    for i in range(1, lagged_terms+1, 1):
        cols.append(dataset.shift(i))
        names += [j+'(t-%s)'%str(i) for j in dataset.columns]

    # To get forecast terms (t, t+1,... t+n used for forecasting)
    for i in range(0, forecast_terms):
        cols.append(dataset.shift(-i))
        if (i==0):
            names += [j+'(t)' for j in dataset.columns]
        else:
            names += [j+'(t+%s)'%str(i) for j in dataset.columns]

    df= pd.DataFrame(pd.concat(cols, axis= 1) )
    df.columns= names
    df.dropna(inplace=True)
    return df


def spierman_correlation(y, X, target_name, predictor_name, alpha=0.05, plot_correlations=True, path_plots=''):
    """
    Takes y (target) and X (set of predictors). Target_name and predictor_name are used for plotting.
    Returns:
    list of features with significant correlation; the feature for which corr is max (with significance > alpha);
    the correlation of this feature and its p value.

    """
    from scipy.stats import spearmanr

    corr_values = []
    p_values = []
    mask = []
    significant_corr = []
    signif_pvalue = []
    feat = []
    for i in X.columns:
        corr, pval = spearmanr(y, X[i])
        corr_values.append(corr)
        p_values.append(pval)

        if (pval < alpha):
            mask.append(1)
            significant_corr.append(corr)
            signif_pvalue.append(pval)
            feat.append(i)
            # print('Feature %s is correlated with target by %g pct of confidence.'%(i, (1-alpha)*100) )
        else:
            mask.append(0)

    corr_values = np.array(corr_values)
    p_values = np.array(p_values)
    mask = np.array(mask)
    significant_corr = np.array(significant_corr)
    signif_pvalue = np.array(signif_pvalue)
    feat = np.array(feat)

    # get argmax value with corresponding pvalue
    lag_term = np.argmax(np.abs(significant_corr))

    if plot_correlations:
        x = np.arange(len(corr_values))
        cindex, ncindex = np.where(mask == 1)[0], np.where(mask == 0)[0]

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.stem(x, corr_values, linefmt='grey')
        # ax.stem(x[cindex], corr_values[cindex], linefmt='grey', markerfmt= 'r', label= 'correlated')
        # if (len(ncindex)!= 0):
        #    ax.stem(x[ncindex], corr_values[ncindex], linefmt='grey', markerfmt= 'b', label = 'uncorrelated')

        ax.set_xlabel('Lagged terms', fontsize=14)
        ax.set_ylabel('Spearman correlation with %s' % target_name, fontsize=14)
        # ax.legend(loc= 'best')
        ax.set_title(predictor_name, fontsize=16)
        plt.savefig(path_plots + predictor_name + '_' + target_name + '.png', bbox_inches='tight')
        plt.close();

    return feat, feat[lag_term], significant_corr[lag_term], signif_pvalue[lag_term]


def split_dataset(df, groups=[], **kwargs):
    dtrain = []
    dtest = []
    for g in groups:
        table = df[df['inlet'] == g].reset_index(drop=True)

        test = table.sample(**kwargs)
        train = table.drop(test.index)

        dtrain.append(train.reset_index(drop=True))
        dtest.append(test.reset_index(drop=True))

    dtrain = pd.concat(dtrain, ignore_index=True)
    dtest = pd.concat(dtest, ignore_index=True)

    return dtrain, dtest

# metrics for regression problems
def mean_absolute_percentage_error(y_true, y_pred, weights=None):
    PE= np.abs((y_true - y_pred) / y_true)*100
    if (weights is not None):
        return np.mean(PE)
    else:
        return np.average(PE, weights= weights)

def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred, weights= None):
    return sklearn.metrics.mean_absolute_error(ytrue, ypred, sample_weight= weights)

def rmse(ytrue, ypred, weights= None):
        return np.sqrt(sklearn.metrics.mean_squared_error(ytrue, ypred,  sample_weight= weights) )

def regression_metrics(ytrue, ypred, weights=None, print_table=True , return_object = True):
    """
    Function that computes the error metrics of a model.
    Input: ytrue -> true values of the target. It can be a pandas series or a numpy array
           ypred -> predictions of the model. Must have the same lenght as ytrue. Series or numpy array.
           weights -> pandas series or numpy array with the weights of the test dataset. Default: None
           print_table -> True if you want a visualisation of the error metrics.
           return_object -> True if you want a dictionary with the error metrics as output.

    Output: pandas dataframe or dictionary with the error metrics; depending on what is set in return_object.
            The error metrics are: MAPE, MdAPE, MAE, RMSE, MdAE, r2, corr, PPE10, PPE30;
            where corr -> is the pearson correlation of ytrue and ypred.
    """
    import sklearn.metrics

    if isinstance(ytrue, pd.Series):
        ytrue = ytrue.values
    if isinstance(ypred, pd.Series):
        ypred = ypred.values

    mape= mean_absolute_percentage_error(ytrue, ypred, weights= weights)
    mdape= median_absolute_percentage_error(ytrue, ypred)
    mae= sklearn.metrics.mean_absolute_error(ytrue, ypred, sample_weight= weights)
    root_mse= rmse(ytrue, ypred, weights= weights)
    mdae= sklearn.metrics.median_absolute_error(ytrue, ypred)
    r2= sklearn.metrics.explained_variance_score(ytrue, ypred, sample_weight= weights)


    metric = pd.DataFrame(data=['%.2f'%mape +'%',
                                '%.2f'%mdape + '%',
                                '%.2f'%mae,
                                '%.2f'%root_mse,
                                '%.2f'%mdae,
                                '%.2f'%r2
                                ],
                            columns = ['Value'],
                            index =['MAPE','MdAPE','MAE','RMSE','Median Absolute Error','Explained Variance']
                            )

    if print_table == True:
        display(metric)

    if return_object == True:
        return {'MAPE': mape, 'MdAPE': mdape, 'MAE': mae, 'RMSE': root_mse, 'MdAE': mdae, 'R2': r2}
    else:
        return metric
