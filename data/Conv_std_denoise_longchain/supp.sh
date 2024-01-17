#!/bin/bash

# moving to source code directory
cd ../../src/ || 
    exit

# data folders
data_dir="../data/Conv_std_denoise_longchain"
num_models=50  # number of models in the chain

# computing distributions from datasets
dset_path="${data_dir}/original_dset-ubyte.gz"
dist_path="${data_dir}/original_dist.png"
python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"

for i in $(seq 0 48); do
    dset_path="${data_dir}/dset_${i}-ubyte.gz"
    dist_path="${data_dir}/dist_${i}.png"
    python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"
done

dset_path="${data_dir}/final_dset-ubyte.gz"
dist_path="${data_dir}/final_dist.png"
python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"

# plotting datasets' difference and variability
python3 plot_variability.py --dset_dir "$data_dir" --num_models "$num_models"
python3 plot_difference.py --dset_dir "$data_dir" --num_models "$num_models"

# moving back to this directory
cd - ||
    exit

exit
