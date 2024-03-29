# this script can be used to train a chain of autoencoders,
# starting from a dataset called 'original_dataset-ubyte.gz'
# located in some directory passed as argument
#
# WARNING: be careful passing command line arguments - grids'
# height and width and dataset size must be coherent with the
# features of the dataset used


from chain_lib import positive_int
from chain_lib import positive_float
from chain_lib import fraction_value
from chain_lib import VAE_FC1
from chain_lib import VAE_FC2
from chain_lib import VAE_Conv
from chain_lib import VAE_Asymm
from chain_lib import train_VAE
from chain_lib import generate_dset
from chain_lib import denoise_dset
import argparse
import time
import torch as th


# default parameters
num_models_dflt = 20  # number of autoencoders in the chain
model_type_dflt = 'Conv'  # autoencoder type
dset_dir_dflt = '../data'  # directory containing data
y_size_dflt = 24  # height of grids
x_size_dflt = 24  # width of grids
dset_size_dflt = 20000  # dataset size
latent_size_dflt = 8  # size of latent space
batch_size_dflt = 32  # size of batch in training
regul_const_dflt = 1.0  # regularization term's multiplicative constant
var_const_dflt = 0.0  # variability term's multiplicative constant
std_increment_dflt = 0.0  # standard deviation increment in generative mode
denoise_thres_dflt = 0.0  # denoising threshold


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=positive_int, default=num_models_dflt)
    parser.add_argument('--model_type', type=str, choices=['FC1', 'FC2', 'Conv', 'Asymm'], default=model_type_dflt)
    parser.add_argument('--dset_dir', type=str, default=dset_dir_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)
    parser.add_argument('--latent_size', type=positive_int, default=latent_size_dflt)
    parser.add_argument('--batch_size', type=positive_int, default=batch_size_dflt)
    parser.add_argument('--regul_const', type=positive_float, default=regul_const_dflt)
    parser.add_argument('--var_const', type=positive_float, default=var_const_dflt)
    parser.add_argument('--std_increment', type=float, default=std_increment_dflt)
    parser.add_argument('--denoise_thres', type=fraction_value, default=denoise_thres_dflt)

    args = parser.parse_args()

    num_models = args.num_models
    model_type = args.model_type
    dset_dir = args.dset_dir
    y_size = args.y_size
    x_size = args.x_size
    dset_size = args.dset_size
    latent_size = args.latent_size
    batch_size = args.batch_size
    regul_const = args.regul_const
    var_const = args.var_const
    std_increment = args.std_increment
    denoise_thres = args.denoise_thres

    # getting device
    device = th.device(device="cuda" if th.cuda.is_available() else "cpu")

    # training the model
    for model_id in range(num_models):

        print(f"\ntraining model number {model_id} of the chain...")

        # instatiating the base model (autoencoder)
        base_model = None
        if model_type == 'FC1':
            base_model = VAE_FC1(y_size, x_size, latent_size, device)
        elif model_type == 'FC2':
            base_model = VAE_FC2(y_size, x_size, latent_size, std_increment, device)
        elif model_type == 'Conv':
            base_model = VAE_Conv(y_size, x_size, latent_size, std_increment, device)
        elif model_type == 'Asymm':
            base_model = VAE_Asymm(y_size, x_size, latent_size, device)

        # getting path to datasets
        input_path = None  # train dataset
        output_path = None  # generated dataset
        if model_id == 0:
            input_path = dset_dir + '/original_dset-ubyte.gz'
            output_path = dset_dir + '/dset_0-ubyte.gz'
        elif model_id == num_models-1:
            input_path = dset_dir + '/dset_' + str(model_id-1) + '-ubyte.gz'
            output_path = dset_dir + '/final_dset-ubyte.gz'
        else:
            input_path = dset_dir + '/dset_' + str(model_id-1) + '-ubyte.gz'
            output_path = dset_dir + '/dset_' + str(model_id) + '-ubyte.gz'

        # measuring iteration's initial time
        start_time = time.perf_counter()

        # training the autoencoder
        base_model = train_VAE(base_model, device, input_path, y_size, x_size, regul_const=regul_const, var_const=var_const, batch_size=batch_size)

        # generating new dataset and writing it to file
        generate_dset(base_model, device, output_path, dset_size)

        # measuring iteration's elapsed time
        elapsed_time = time.perf_counter() - start_time

        print(f"model trained and dataset generated\n\ttotal iteration time: {elapsed_time} seconds")

        # denoising output dataset (optional)
        if denoise_thres:
            denoise_dset(output_path, dset_size, y_size, x_size, denoise_thres)
