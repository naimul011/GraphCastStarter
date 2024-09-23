import cdsapi
import datetime
import functools
from graphcast import autoregressive, casting, checkpoint, data_utils as du, graphcast, normalization, rollout
import haiku as hk
import isodate
import jax
import math
import numpy as np
import pandas as pd
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude
import pytz
import scipy
from typing import Dict
import xarray

print('Libraries Imported!!')

from utils.params import *


print('Parameters and Assumptions')

url = ''
key = ''

# Create the CDS API client instance with the specified configuration file
client = cdsapi.Client(url=url, key=key)

# Now you can use the client to fetch data
print('Now you can use the client to fetch data!!!')

client = cdsapi.Client() # Making a connection to CDS, to fetch data.


print('Assumptions and important parameters!!')

# creating the input data
print('Creating the input data!!!')
# Getting the single and pressure level values.
def getSingleAndPressureValues():
    
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': singlelevelfields,
            'grid': '1.00/1.00',
            'year': [2024],
            'month': [1],
            'day': [2],
            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00'],
            'format': 'netcdf'
        },
        'single-level.nc'
    )
    singlelevel = xarray.open_dataset('single-level.nc', engine = scipy.__name__).to_dataframe()
    singlelevel = singlelevel.rename(columns = {col:singlelevelfields[ind] for ind, col in enumerate(singlelevel.columns.values.tolist())})
    singlelevel = singlelevel.rename(columns = {'geopotential': 'geopotential_at_surface'})

    # Calculating the sum of the last 6 hours of rainfall.
    singlelevel = singlelevel.sort_index()
    singlelevel['total_precipitation_6hr'] = singlelevel.groupby(level=[0, 1])['total_precipitation'].rolling(window = 6, min_periods = 1).sum().reset_index(level=[0, 1], drop=True)
    singlelevel.pop('total_precipitation')
    
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': pressurelevelfields,
            'grid': '1.0/1.0',
            'year': [2024],
            'month': [1],
            'day': [2],
            'time': ['00:00','06:00', '12:00','18:00'],
            'pressure_level': pressure_levels,
            'format': 'netcdf'
        },
        'pressure-level.nc'
    )
    pressurelevel = xarray.open_dataset('pressure-level.nc', engine = scipy.__name__).to_dataframe()
    pressurelevel = pressurelevel.rename(columns = {col:pressurelevelfields[ind] for ind, col in enumerate(pressurelevel.columns.values.tolist())})

    return singlelevel, pressurelevel

# Adding sin and cos of the year progress.
def addYearProgress(secs, data):

    progress = du.get_year_progress(secs)
    data['year_progress_sin'] = math.sin(2 * pi * progress)
    data['year_progress_cos'] = math.cos(2 * pi * progress)

    return data

# Adding sin and cos of the day progress.
def addDayProgress(secs, lon:str, data:pd.DataFrame):

    lons = data.index.get_level_values(lon).unique()
    progress:np.ndarray = du.get_day_progress(secs, np.array(lons))
    prxlon = {lon:prog for lon, prog in list(zip(list(lons), progress.tolist()))}
    data['day_progress_sin'] = data.index.get_level_values(lon).map(lambda x: math.sin(2 * pi * prxlon[x]))
    data['day_progress_cos'] = data.index.get_level_values(lon).map(lambda x: math.cos(2 * pi * prxlon[x]))
    
    return data

# Adding day and year progress.
def integrateProgress(data:pd.DataFrame):
        
    for dt in data.index.get_level_values('time').unique():
        seconds_since_epoch = toDatetime(dt).timestamp()
        data = addYearProgress(seconds_since_epoch, data)
        data = addDayProgress(seconds_since_epoch, 'longitude' if 'longitude' in data.index.names else 'lon', data)

    return data

# Adding batch field and renaming some others.
def formatData(data:pd.DataFrame) -> pd.DataFrame:
        
    data = data.rename_axis(index = {'latitude': 'lat', 'longitude': 'lon'})
    if 'batch' not in data.index.names:
        data['batch'] = 0
        data = data.set_index('batch', append = True)
    
    return data

# Includes the packages imported and constants assigned.
# The functions created for the inputs also go here.

predictionFields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'mean_sea_level_pressure',
                        'total_precipitation_6hr'
                    ]

# Creating an array full of nan values.
def nans(*args) -> list:
    return np.full((args), np.nan)

# Adding or subtracting time.
def deltaTime(dt, **delta) -> datetime.datetime:
    return dt + datetime.timedelta(**delta)

def getTargets(dt, data:pd.DataFrame):
    
    # Creating an array consisting of unique values of each index.
    lat, lon, levels, batch = sorted(data.index.get_level_values('lat').unique().tolist()), sorted(data.index.get_level_values('lon').unique().tolist()), sorted(data.index.get_level_values('level').unique().tolist()), data.index.get_level_values('batch').unique().tolist()
    time = [deltaTime(dt, hours = days * gap) for days in range(4)]

    # Creating an empty dataset using latitude, longitude, the pressure levels and each prediction timestamp.
    target = xarray.Dataset({field: (['lat', 'lon', 'level', 'time'], nans(len(lat), len(lon), len(levels), len(time))) for field in predictionFields}, coords = {'lat': lat, 'lon': lon, 'level': levels, 'time': time, 'batch': batch})

    return target.to_dataframe()

# Includes the packages imported and constants assigned.
# The functions created for the inputs and targets also go here.

# Adding a timezone to datetime.datetime variables.
def addTimezone(dt, tz = pytz.UTC) -> datetime.datetime:
    dt = toDatetime(dt)
    if dt.tzinfo == None:
        return pytz.UTC.localize(dt).astimezone(tz)
    else:
        return dt.astimezone(tz)

# Getting the solar radiation value wrt longitude, latitude and timestamp.
def getSolarRadiation(longitude, latitude, dt):
        
    altitude_degrees = get_altitude(latitude, longitude, addTimezone(dt))
    solar_radiation = get_radiation_direct(dt, altitude_degrees) if altitude_degrees > 0 else 0

    return solar_radiation * watts_to_joules

# Calculating the solar radiation values for timestamps to be predicted.
def integrateSolarRadiation(data:pd.DataFrame):
    
    dates = list(data.index.get_level_values('time').unique())
    coords = [[lat, lon] for lat in lat_range for lon in lon_range]
    values = []
    
    # For each data, getting the solar radiation value at a particular coordinate.
    for dt in dates:
        values.extend(list(map(lambda coord:{'time': dt, 'lon': coord[1], 'lat': coord[0], 'toa_incident_solar_radiation': getSolarRadiation(coord[1], coord[0], dt)}, coords)))
  
    # Setting indices.
    values = pd.DataFrame(values).set_index(keys = ['lat', 'lon', 'time'])
      
    # The forcings dataset will now contain the solar radiation values.
    return pd.merge(data, values, left_index = True, right_index = True, how = 'inner')

def getForcings(data:pd.DataFrame):
  
    # Since forcings data does not contain batch as an index, it is dropped.
    # So are all the columns, since forcings data only has 5, which will be created.
    forcingdf = data.reset_index(level = 'level', drop = True).drop(labels = predictionFields, axis = 1)
    
    # Keeping only the unique indices.
    forcingdf = pd.DataFrame(index = forcingdf.index.drop_duplicates(keep = 'first'))

    # Adding the sin and cos of day and year progress.
    # Functions are included in the creation of inputs data section.
    forcingdf = integrateProgress(forcingdf)

    # Integrating the solar radiation values.
    forcingdf = integrateSolarRadiation(forcingdf)

    return forcingdf

# Post-processing the inputs, targets and forcings
# Includes the packages imported and constants assigned.
# The functions created for the inputs, targets and forcings also go here.

# A dictionary created, containing each coordinate a data variable requires.
class AssignCoordinates:
    
    coordinates = {
                    '2m_temperature': ['batch', 'lon', 'lat', 'time'],
                    'mean_sea_level_pressure': ['batch', 'lon', 'lat', 'time'],
                    '10m_v_component_of_wind': ['batch', 'lon', 'lat', 'time'],
                    '10m_u_component_of_wind': ['batch', 'lon', 'lat', 'time'],
                    'total_precipitation_6hr': ['batch', 'lon', 'lat', 'time'],
                    'temperature': ['batch', 'lon', 'lat', 'level', 'time'],
                    'geopotential': ['batch', 'lon', 'lat', 'level', 'time'],
                    'u_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
                    'v_component_of_wind': ['batch', 'lon', 'lat', 'level', 'time'],
                    'vertical_velocity': ['batch', 'lon', 'lat', 'level', 'time'],
                    'specific_humidity': ['batch', 'lon', 'lat', 'level', 'time'],
                    'toa_incident_solar_radiation': ['batch', 'lon', 'lat', 'time'],
                    'year_progress_cos': ['batch', 'time'],
                    'year_progress_sin': ['batch', 'time'],
                    'day_progress_cos': ['batch', 'lon', 'time'],
                    'day_progress_sin': ['batch', 'lon', 'time'],
                    'geopotential_at_surface': ['lon', 'lat'],
                    'land_sea_mask': ['lon', 'lat'],
                }

def modifyCoordinates(data:xarray.Dataset):
    
    # Parsing through each data variable and removing unneeded indices.
    for var in list(data.data_vars):
        varArray:xarray.DataArray = data[var]
        nonIndices = list(set(list(varArray.coords)).difference(set(AssignCoordinates.coordinates[var])))
        data[var] = varArray.isel(**{coord: 0 for coord in nonIndices})
    data = data.drop_vars('batch')

    return data

def makeXarray(data:pd.DataFrame) -> xarray.Dataset:
    
    # Converting to xarray.
    data = data.to_xarray()
    data = modifyCoordinates(data)

    return data


# Predictions using Graphcast
# Includes the packages imported and constants assigned.
# The functions created for the inputs, targets and forcings also go here.

with open(r'GraphCast/model/params/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz', 'rb') as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

with open(r'GraphCast/model/stats/stats_diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()

with open(r'GraphCast/model/stats/stats_mean_by_level.nc', 'rb') as f:
    mean_by_level = xarray.load_dataset(f).compute()

with open(r'GraphCast/model/stats/stats_stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xarray.load_dataset(f).compute()
    
def construct_wrapped_graphcast(model_config:graphcast.ModelConfig, task_config:graphcast.TaskConfig):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level = diffs_stddev_by_level, mean_by_level = mean_by_level, stddev_by_level = stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing = True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template = targets_template, forcings = forcings)

def with_configs(fn):
    return functools.partial(fn, model_config = model_config, task_config = task_config)

def with_params(fn):
    return functools.partial(fn, params = params, state = state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

class Predictor:

    @classmethod
    def predict(cls, inputs, targets, forcings) -> xarray.Dataset:
        predictions = rollout.chunked_prediction(run_forward_jitted, rng = jax.random.PRNGKey(0), inputs = inputs, targets_template = targets, forcings = forcings)
        return predictions


if __name__ == '__main__':

    values:Dict[str, xarray.Dataset] = {}
    
    single, pressure = getSingleAndPressureValues()
    values['inputs'] = pd.merge(pressure, single, left_index = True, right_index = True, how = 'inner')
    values['inputs'] = integrateProgress(values['inputs'])
    values['inputs'] = formatData(values['inputs'])

    # The code for creating inputs will be here.

    values['targets'] = getTargets(first_prediction, values['inputs'])
    values['forcings'] = getForcings(values['targets'])
    values = {value:makeXarray(values[value]) for value in values}


    print('Inputs: ')
    print(values)
    values.to_dataframe().to_csv('inputs.csv', sep = ',')

    #predictions = Predictor.predict(values['inputs'], values['targets'], values['forcings'])
    #predictions.to_dataframe().to_csv('predictionsMar13-2.csv', sep = ',')

    #print('Predictions: ')
    #print(predictions)
