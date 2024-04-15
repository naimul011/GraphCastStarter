# @title Imports

import dataclasses

import functools

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


# Load weather data
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

plot_diagram.plot_data(data, fig_title, plot_size, plot_example_robust,filename='geopotential_at_surface_Mar1')
print('Ploting done!! Go to GraphCast/output!! ')

# @title Choose training and eval data to extract
# 1 - 3
train_steps = 1
eval_steps = 1

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

print('Extract training and eval data!!!')


# @title Load normalization data

with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()

print('Load normalization data!!')

# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

print('Build jitted functions, and possibly initialize random weights!!!')

# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
predictions

print(predictions)
print('Autoregressive rollout (loop in python)!!!')

# @title Choose predictions to plot

# plot_example_variable(options=('geopotential_at_surface', 'land_sea_mask', '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'toa_incident_solar_radiation', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity'), value='2m_temperature'))
# plot_example_level(options=(1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000), value=500)
# plot_example_robust(value=True, description='Robust')
# plot_example_max_steps(IntSlider(value=3, description='Max steps', max=3, min=1))

# Plot example data
# @title Plot example data

# plot_size = 7

# plot_example_variable = 'geopotential_at_surface'
# plot_example_level = 500
# plot_example_max_steps = 3
# plot_example_robust = True

plot_pred_variable = "2m_temperature"
plot_pred_level = 500
plot_pred_robust = True
plot_pred_max_steps = 3

plot_size = 5
plot_max_steps = 1

data = {
    "Targets": plot_diagram.scale(plot_diagram.select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
    "Predictions": plot_diagram.scale(plot_diagram.select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
    "Diff": plot_diagram.scale((plot_diagram.select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps) -
                        plot_diagram.select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps)),
                       robust=plot_pred_robust, center=0),
}
fig_title = plot_pred_variable
if "level" in predictions[plot_pred_variable].coords:
  fig_title += f" at {plot_pred_level} hPa"

plot_diagram.plot_data(data, fig_title, plot_size, plot_pred_robust)

print('Plot predictions!!!')

predictions.to_dataframe().to_csv('predictionsMar13.csv', sep = ',')