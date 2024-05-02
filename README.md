In this paper, the authors propose a new approach to asset pricing models using latent factors such as PCA.
They develop improved autoencoders whose architecture is based on the combination of two neural networks.
A first neural network is created without a hidden layer and takes as input parameter the asset returns
at each time step, deduced from the risk-free rate, to form K latent factors (V).
The second neural network is trained with several hidden layers and has as input asset characteristics such as risk and trend indicators, in order to deduce the exposure of assets to these systematic factors (factor loadings -> W).
Finally, the outputs of these two networks are multiplied to create the return predictions deduced from the risk-free rate.
The function to be maximized is the R2 score between initial and predicted returns.

I implemented the various steps of the paper using PyTorch and came up with results close to those of the authors, showing that autoencoders perform better than conventional models such as PCA or the Fama and French factor model.

This better performance can be deduced from the fact that applying non-linear transformations to asset characteristics enables the extraction of complex patterns and the construction of highly informative factor exposures.
