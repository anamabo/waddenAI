import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import sklearn.metrics

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

def regression_metrics(ytrue, ypred, weights=None):
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

    return metric



h2o.init(ip="localhost", port=54323, nthreads=3)

train = pd.read_csv('../../2_data/enriched_data/model3_trainsampled.csv.gz')
#test = pd.read_csv('../../2_data/enriched_data/model3_test.csv.gz')
print(train.shape)
#train = train.iloc[0:1000, :]
#test = test.iloc[0:1000, :]

print('FINISHED READING THE DATASETS.')

train1 = h2o.H2OFrame(train)
#test1 = h2o.H2OFrame(test)

print('Starting autoML process:')
# Select the columns used for modelling
columns = ['chl_NN', 'salinity', 'surge','month', 'autumn', 'spring', 'summer', 'winter',
          'pct_mud', 'pct_sand','bathymetry', 'max_wind_speed_1_days_ago',
       'avg_wind_speed_1_days_ago', 'dir_max_wind_vel_1_days_ago',
       'max_wind_speed_2_days_ago', 'avg_wind_speed_2_days_ago',
       'dir_max_wind_vel_2_days_ago', 'max_wind_speed_3_days_ago',
       'avg_wind_speed_3_days_ago', 'dir_max_wind_vel_3_days_ago',
       'max_wind_speed_4_days_ago', 'avg_wind_speed_4_days_ago',
       'dir_max_wind_vel_4_days_ago', 'max_wind_speed_5_days_ago',
       'avg_wind_speed_5_days_ago', 'dir_max_wind_vel_5_days_ago',
       'max_wind_speed_6_days_ago', 'avg_wind_speed_6_days_ago',
       'dir_max_wind_vel_6_days_ago', 'max_wind_speed_7_days_ago',
       'avg_wind_speed_7_days_ago', 'dir_max_wind_vel_7_days_ago',
       'max_wind_speed_8_days_ago', 'avg_wind_speed_8_days_ago',
       'dir_max_wind_vel_8_days_ago', 'Ame_binnen', 'Ame_buiten',
       'Eems_binnen', 'Eems_buiten', 'Eld_binnen', 'Eld_buiten', 'Frz_binnen',
       'Frz_buiten', 'Md_binnen', 'Md_buiten', 'Vlie_binnen', 'Vlie_buiten']
target = 'spm'


# Run AutoML for 15 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models= 15,
                nfolds= 5,
                stopping_metric= 'RMSE',
                sort_metric= 'RMSE',
                seed=12345
               )

aml.train(y= target, x= columns, training_frame=train1)

print('AutoML process finished.')

lb = aml.leaderboard.as_data_frame()
lb.to_csv('../../5_models/waddenAI/model3/leader_board.csv', index= False)
print( lb.head())

# https://stackoverflow.com/questions/55081358/how-to-find-best-params-of-leader-model-in-automl-h2o-python
# Get model ids for all models in the AutoML Leaderboard

for model in lb['model_id']:
    m = h2o.get_model(model)
    bla = h2o.save_model(model=m, path="../../5_models/waddenAI/model3/", force=True)


# Get final predictions on test set
bestmod_noens= h2o.get_model(lb['model_id'][3])

ypred = bestmod_noens.predict(test1).as_data_frame()
ypred.rename(columns= {'predict': 'predictions'}, inplace= True)

test = pd.merge(test,ypred, left_index= True, right_index= True)

# Get final error metrics
metrics = regression_metrics(test[target], test['predictions'])
metrics.to_csv("../../5_models/waddenAI/model4/metrics_model4.csv")
