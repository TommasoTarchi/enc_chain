# Enc\_Chain


***Work in progress...***


## What is this all about

The aim of the project will be that of investigating the effect of training
variational autoencoders on synthetic datasets.

This will be achieved by building a *chain* of variational autoencoders, each 
one trained on a dataset generated by the previous one.
The first autoencoder of the chain will be trained on an "ad hoc" dataset made
of grids in which pixels (actually a square cluster of seven pixels) are turned
on according to a given distribution.

The outcome will be interesting (**in my opinion**) because this is just a very 
simple and particular example of a much broader topic, that is AI models
trained on AI-generated data.

Whether the project's aim makes sense or not is highly debatable (I still have
to figure it out).


## What you will find in this repository

The repo is structured in the following way:

- This README file
- `src/`: directory containing all source code used for the project; contains:
  - `chain_lib.py`: library containing all classes and functions used in the 
  scripts
  - `gen_dataset.py`: script that can be used to generate a syntetic dataset 
  with pixels turned on according to some (chosen) distribution
  - `make_chain.py`: script that can be used to run a chain of VAEs (of variable
  length) starting from a target dataset
  - `comp_distribution.py`: script that can be used to compute the distribution 
  of the turned on pixels in some dataset
  - `show_grids.ipynb`: notebook that can be used to show a certain number of
  random images from a given dataset (used to check that the autoencoders are 
  working as expected)
- `data/`: directory containing data gathered with different parameters


## General pipeline

All data gathered for this project were obtained with the following general 
procedure:

1. Create an empty directory inside `data/` to store initial, final and 
intermediate datasets, with related distributions
2. Inside the directory created at point 1., create an initial dataset called 
`original_dset-ubyte.gz` using the `gen_dataset.py` script (set the desired 
parameters by using the script's command line arguments)
3. Run the chain using `make_chain.py`, setting the desired command line arguments
(**notice** that the arguments passed to this script must be coherent to the ones
used to generate the dataset at step 3., and that the path to the directory to 
store datasets must be passed from command line as well)
4. For all datasets produced by the chain (saved in the directory created at point
1.), compute the related distribution of turned on pixels using 
`comp_distribution.py`, remembering to pass as command line arguments both the
path to dataset and the path to where you want the related distribution's plot to
be stored.


## Reproducing results

To reproduce the data used in the project you can:

1. Navigate to the data folder corresponding to the data you want to reproduce
2. Run `bash run.sh`; the Bash script wil produce an initial dataset called 
`original_dset.gz` (and corresponding pixels distribution `original_dist.png`),
the intermediate datasets called `dset_$i.gz` (and corresponding pixels
distributions`dist_$i.png`), where `$i` is the index of the autoencoder in the 
chain used to generate the dataset, and the final dataset called `final_dset.gz` 
(and corresponding pixels distribution `final_dist.png`)
