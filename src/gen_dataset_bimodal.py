# this script can be used to generate the synthetic dataset
# to train the first encoder of the chain
#
# two-peaks distribution


from chain_lib import positive_int
import argparse
import gzip
import numpy as np


# default parameters
dset_path_dflt = '../data/original_dset-ubyte.gz'  # path to save data
dset_size_dflt = 20000  # size of the dataset
y_size_dflt = 24  # height of the grids
x_size_dflt = 24  # width of the grids
y_dist_dflt = 'binomial'  # distribution of the y coordinates
x_dist_dflt = 'binomial'  # distribution of the x coordinates


if __name__ == "__main__":

    # getting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_path', type=str, default=dset_path_dflt)
    parser.add_argument('--dset_size', type=positive_int, default=dset_size_dflt)
    parser.add_argument('--y_size', type=positive_int, default=y_size_dflt)
    parser.add_argument('--x_size', type=positive_int, default=x_size_dflt)
    parser.add_argument('--y_dist', choices=['binomial', 'uniform'], default=y_dist_dflt)
    parser.add_argument('--x_dist', choices=['binomial', 'uniform'], default=x_dist_dflt)

    args = parser.parse_args()

    dset_path = args.dset_path
    dset_size = args.dset_size
    y_size = args.y_size
    x_size = args.x_size
    y_dist = args.y_dist
    x_dist = args.x_dist

    # initializing the dataset
    grids = np.zeros((dset_size, y_size, x_size), dtype=np.ubyte)

    # extracting the y coordinates
    y_coord = np.empty(dset_size)
    if y_dist == 'binomial':
        success_prob = np.empty(dset_size)
        success_prob[:int(dset_size*2/3)] = 0.8
        success_prob[int(dset_size*2/3):] = 0.2
        y_coord = np.random.binomial(y_size-1, success_prob, dset_size)
    elif y_dist == 'uniform':
        y_coord = np.random.randint(0, y_size, dset_size)

    # extracting the x coordinates
    x_coord = np.empty(dset_size)
    if x_dist == 'binomial':
        success_prob = np.empty(dset_size)
        success_prob[:int(dset_size*2/3)] = 0.3
        success_prob[int(dset_size*2/3):] = 0.7
        x_coord = np.random.binomial(x_size-1, success_prob, dset_size)
    elif x_dist == 'uniform':
        x_coord = np.random.randint(0, x_size, dset_size)

    # randomly turning on one pixel and 7 neighbour ones
    for i in range(grids.shape[0]):
        grids[i][y_coord[i]][x_coord[i]] = 255

        # pixels on the top row
        if y_coord[i] == 0:
            if x_coord[i] != 0:
                grids[i][y_coord[i]][x_coord[i]-1] = 255
                grids[i][y_coord[i]+1][x_coord[i]-1] = 255
            if x_coord[i] != x_size-1:
                grids[i][y_coord[i]][x_coord[i]+1] = 255
                grids[i][y_coord[i]+1][x_coord[i]+1] = 255
            grids[i][y_coord[i]+1][x_coord[i]] = 255

        # pixels on the bottom row
        elif y_coord[i] == y_size-1:
            if x_coord[i] != 0:
                grids[i][y_coord[i]-1][x_coord[i]-1] = 255
                grids[i][y_coord[i]][x_coord[i]-1] = 255
            if x_coord[i] != x_size-1:
                grids[i][y_coord[i]-1][x_coord[i]+1] = 255
                grids[i][y_coord[i]][x_coord[i]+1] = 255
            grids[i][y_coord[i]-1][x_coord[i]] = 255

        # pixels on middle rows
        else:
            if x_coord[i] != 0:
                grids[i][y_coord[i]-1][x_coord[i]-1] = 255
                grids[i][y_coord[i]][x_coord[i]-1] = 255
                grids[i][y_coord[i]+1][x_coord[i]-1] = 255
            if x_coord[i] != x_size-1:
                grids[i][y_coord[i]-1][x_coord[i]+1] = 255
                grids[i][y_coord[i]][x_coord[i]+1] = 255
                grids[i][y_coord[i]+1][x_coord[i]+1] = 255
            grids[i][y_coord[i]-1][x_coord[i]] = 255
            grids[i][y_coord[i]+1][x_coord[i]] = 255

    # reshuffling dataset
    indices = np.arange(dset_size)
    np.random.shuffle(indices)
    grids = grids[indices]

    # loading dataset to file
    grids_byte = grids.copy().tobytes()
    with gzip.open(dset_path, 'wb') as f:
        f.write(grids_byte)
