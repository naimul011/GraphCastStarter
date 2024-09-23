# GraphCastStarter

This repository provides guidance on running the GraphCast weather forecasting model on a remote GPU server.


![alt text]([http://url/to/img.png](https://lh3.googleusercontent.com/xJi_k_ZjLpJNgfrcWGxSDtxtv_ZA17qvsYzqkl55jWM4sTLBLfO1XVKIBoMqnTk-1fSnxs58KssQuSJLhVXrFeS8FSz0RJ2bnITszAi67T5jDW4ttQ=w1072-h603-n-nu-rw)) 
## Setup

1. Connect to the remote server:
   - Use PuTTY or SSH via VSCode
   - Server: login.delta.ncsa.illinois.edu

2. Environment setup:
   ```bash
   conda env create -f environment.yml
   source activate naim
   ```

3. Install additional dependencies:
   ```bash
   pip install google-cloud google-cloud-vision
   conda install -c conda-forge python=3.10 -y
   ```

## Running GraphCast

Use the provided `jobfile.sh` script to run GraphCast:

```bash
sbatch jobfile.sh
```

Modify `jobfile.sh` to adjust GPU options and job parameters as needed.

### Training

To train GraphCast, you'll need to download the ERA5 dataset. Refer to the [Weatherbench2 ERA5 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5) for instructions.

### Prediction

For prediction, use the pre-trained models available on the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

### Running the Demo

Open `graphcast_demo.ipynb` in [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb) for an interactive demonstration of loading data, generating predictions, and computing gradients.

## Troubleshooting

- If you encounter module import errors, ensure all dependencies are correctly installed in your conda environment.
- For permission issues, avoid running `setup.py` directly. Use the provided conda environment instead.

## Monitoring Jobs

Use the OnDemand platform at delta.ncsa.illinois.edu to view job status and manage running jobs.

## License

This project is licensed under the Apache License, Version 2.0. The model weights are available under the CC BY-NC-SA 4.0 license.

For more detailed information, refer to the original [GraphCast repository](https://github.com/deepmind/graphcast).
