#!/bin/bash


##SBATCH -e "errFile"$1".txt"
#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1    # match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 # or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account= #GPU 
#SBATCH --job-name=myjobtest
#SBATCH --time=00:05:00      # hh:mm:ss for the job
#SBATCH --output="/u/msoltaninia/VQS_pennylane/$2/$1(qubits)-$2.$random_number.out"
##SBATCH --output=${output_name}






### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # or closest
##SBATCH --mail-user=nh16@alfred.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options



module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"

# source activate global_finder310
conda create --name naim

# pip install matplotlib
# pip install kahypar
# pip install scipy
# pip install pennylane-sparse
# conda install -c conda-forge cuquantum-python
# conda install -c "conda-forge/label/broken" cuquantum-python
# pip install pennylane-lightning[gpu]

# conda remove -c anaconda cudatoolkit -y
# conda clean --all
# conda install -c anaconda cudatoolkit
# pip install pennylane[torch,cirq,qiskit]
#pip install qiskit
#pip install qiskit-aer-gpu
#pip install qiskit-aqua

# conda install -c conda-forge custatevec
# conda install -c conda-forge cuquantum
# export CUQUANTUM_ROOT=$CONDA_PREFIX
#pip install pennylane-cirq
# pip install qsimcirq
#pip install pennylane_pq
#pip install pennylane-qsharp
# srun python3 runner.py $1 $2
echo "done"