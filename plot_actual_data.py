# @title Imports

import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import os
import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
from utils import plot_diagram

print('Library imported!!')

# @title Authenticate with Google Cloud Storage

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")

print('Authenticate with Google Cloud Storage!!!')

# ['source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc', 'source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc', 'source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc', 'source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc', 'source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc', 'source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc', 'source-era5_date-2022-01-01_res-1.0_levels-13_steps-20.nc', 'source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc', 'source-era5_date-2022-01-01_res-1.0_levels-37_steps-01.nc', 'source-era5_date-2022-01-01_res-1.0_levels-37_steps-04.nc', 'source-era5_date-2022-01-01_res-1.0_levels-37_steps-12.nc', 'source-era5_date-2022-01-01_res-1.0_levels-37_steps-20.nc', 'source-era5_date-2022-01-01_res-1.0_levels-37_steps-40.nc', 'source-fake_date-2000-01-01_res-6.0_levels-13_steps-01.nc', 'source-fake_date-2000-01-01_res-6.0_levels-13_steps-04.nc', 'source-fake_date-2000-01-01_res-6.0_levels-13_steps-12.nc', 'source-fake_date-2000-01-01_res-6.0_levels-13_steps-20.nc', 'source-fake_date-2000-01-01_res-6.0_levels-13_steps-40.nc', 'source-fake_date-2000-01-01_res-6.0_levels-37_steps-01.nc', 'source-fake_date-2000-01-01_res-6.0_levels-37_steps-04.nc', 'source-fake_date-2000-01-01_res-6.0_levels-37_steps-12.nc', 'source-fake_date-2000-01-01_res-6.0_levels-37_steps-20.nc', 'source-fake_date-2000-01-01_res-6.0_levels-37_steps-40.nc', 'source-hres_date-2022-01-01_res-0.25_levels-13_steps-01.nc', 'source-hres_date-2022-01-01_res-0.25_levels-13_steps-04.nc', 'source-hres_date-2022-01-01_res-0.25_levels-13_steps-12.nc']
params_file = 'GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'

with gcs_bucket.blob(f"params/{params_file}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

print(model_config)
print('Load the model!!!')

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

# Dropdown(description='Dataset file:', layout=Layout(width='max-content'), options=(('source: era5, date: 2022-01-01, res: 0.25, levels: 37, steps: 01', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc'), ('source: era5, date: 2022-01-01, res: 0.25, levels: 37, steps: 04', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc'), ('source: era5, date: 2022-01-01, res: 0.25, levels: 37, steps: 12', 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc')), value='source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc')
dataset_file = 'source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc'

with gcs_bucket.blob(f"dataset/{dataset_file}").open("rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))

print(example_batch)
print('Load weather data!!!')

# Choose data to plot

# plot_example_variable(options=('geopotential_at_surface', 'land_sea_mask', '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'toa_incident_solar_radiation', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity'), value='2m_temperature'))
# plot_example_level(options=(1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000), value=500)
# plot_example_robust(value=True, description='Robust')
# plot_example_max_steps(IntSlider(value=3, description='Max steps', max=3, min=1))

# Plot example data
# @title Plot example data

plot_size = 7

plot_example_variable = 'geopotential_at_surface'
plot_example_level = 500
plot_example_max_steps = 3
plot_example_robust = True

data = {
    " ": plot_diagram.scale(plot_diagram.select(example_batch, plot_example_variable, plot_example_level, plot_example_max_steps),
              robust=plot_example_robust),
}
fig_title = plot_example_variable
if "level" in example_batch[plot_example_variable].coords:
  fig_title += f" at {plot_example_level} hPa"

plot_diagram.plot_data(data, fig_title, plot_size, plot_example_robust,filename='geopotential_at_surface_Mar_1')
print('Ploting done!! Go to GraphCast/output!! ')