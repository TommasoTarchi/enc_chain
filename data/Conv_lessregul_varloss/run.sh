#!/bin/bash

# cleaning current directory
rm -rf *.gz
rm -rf *.png
rm -rf *.out

# moving to source code directory
cd ../../src/ || 
    exit

# data folders
data_dir="../data/Conv_lessregul_varloss"
base_model=Conv
regularization=0.7  # regularization constant for the loss function
variability=10.0  # multiplicative constant of the variability term in loss

# generating initial dataset
python3 gen_dataset.py --dset_path "${data_dir}/original_dset-ubyte.gz"

# running autoencoders chain
python3 make_chain.py --dset_dir "$data_dir" --model_type "$base_model" --regul_const "$regularization" --var_const "$variability"  >  "${data_dir}/report.out"

# computing distributions from datasets
dset_path="${data_dir}/original_dset-ubyte.gz"
dist_path="${data_dir}/original_dist.png"
python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"

for i in $(seq 0 18); do
    dset_path="${data_dir}/dset_${i}-ubyte.gz"
    dist_path="${data_dir}/dist_${i}.png"
    python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"
done

dset_path="${data_dir}/final_dset-ubyte.gz"
dist_path="${data_dir}/final_dist.png"
python3 comp_distribution.py --dset_path "$dset_path" --plot_path "$dist_path"

# plotting datasets' difference and variability
python3 plot_variability.py --dset_dir "$data_dir"
python3 plot_difference.py --dset_dir "$data_dir"

# moving back to this directory
cd - ||
    exit

exit
