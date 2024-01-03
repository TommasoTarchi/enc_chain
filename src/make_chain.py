# this script can be used to train a chain of autoencoders,
# starting from a dataset called 'original_dataset-ubyte.gz'
# located in some directory passed as argument
#
# WARNING: be careful passing command line arguments - grids'
# height and width and dataset size must be coherent with the
# features of the dataset used


import argparse
import time
from chain_lib import positive_int
from chain_lib import FC_AutoEncoder
from chain_lib import Conv_AutoEncoder
from chain_lib import train_AutoEncoder
from chain_lib import generate_dset
import torch as th


# default parameters
num_models_dflt = 20  # number of autoencoders in the chain
model_type_dflt = 'FC'  # autoencoder type
dset_dir_dflt = '../data/'  # directory containing data
y_size_dflt = 30  # height of grids
x_size_dflt = 30  # width of grids
dset_size_dflt = 20000  # dataset size
latent_size_dflt = 8  # size of latent space


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=positive_int, default=num_models_dflt)
    parser.add_argument('--model_type', type=str, choices=['FC', 'conv'], default=model_type_dflt)
    parser.add_argument('--dset_dir', type=str, default=dset_dir_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)
    parser.add_argument('--latent_size', type=positive_int, default=latent_size_dflt)

    args = parser.parse_args()

    num_models = args.num_models
    model_type = args.model_type
    dset_dir = args.dset_dir
    y_size = args.y_size
    x_size = args.x_size
    dset_size = args.dset_size
    latent_size = args.latent_size

    # getting device
    device = th.device(device="cuda" if th.cuda.is_available() else "cpu")

    # training the model
    for model_id in range(num_models):

        print(f"\ntraining model number {model_id} of the chain...")

        # instatiating the base model (autoencoder)
        base_model = None
        if model_type == 'FC':
            base_model = FC_AutoEncoder(y_size, x_size, latent_size, device)
        elif model_type == 'conv':
            base_model = Conv_AutoEncoder(y_size, x_size, latent_size, device)

        # getting path to datasets
        input_path = None  # train dataset
        output_path = None  # generated dataset
        if model_id == 0:
            input_path = dset_dir + 'original_dataset-ubyte.gz'
            output_path = dset_dir + 'dataset_0-ubyte.gz'
        elif model_id == num_models-1:
            input_path = dset_dir + 'dataset_' + str(model_id-1) + '-ubyte.gz'
            output_path = dset_dir + 'final_dataset-ubyte.gz'
        else:
            input_path = dset_dir + 'dataset_' + str(model_id-1) + '-ubyte.gz'
            output_path = dset_dir + 'dataset_' + str(model_id) + '-ubyte.gz'

        # measuring iteration's initial time
        start_time = time.perf_counter()

        # training the autoencoder
        base_model = train_AutoEncoder(base_model, device, input_path, y_size, x_size)

        # generating new dataset and writing it to file
        generate_dset(base_model, device, output_path, dset_size)

        # measuring iteration's elapsed time
        elapsed_time = time.perf_counter() - start_time

        print(f"model trained and dataset generated\n\ttotal iteration time: {elapsed_time} seconds")
