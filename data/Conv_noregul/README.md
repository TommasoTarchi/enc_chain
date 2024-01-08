## Convolutional NN architecture - NO regularization

In this directory you can find data gathered using a 
convolutional architecture for both encoder and decoder of 
each VAE of the chain, and **completely switching off** the
regularization term (i.e the regularization constant is set 
to 0) for the loss function used in the training.

All other parameters are set to default (in particular the 
coordinates of the pixels turned on in the original dataset 
are distributed according to binomial distributions).

**Actually**, we expect the VAEs in this chain to not work at
all, since the distribution in the latent space will be 
unknown (and most likely not a standard normal).
