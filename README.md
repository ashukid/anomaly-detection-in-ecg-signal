## Implementation of the paper " Regularised Encoder-Decoder Architecture for Anomaly Detection in ECG time signal "

This repository contains implementation of the method proposed in the above mentioned paper.

Brief summary from the paper :
```Normal encoder-decoder with just reconstruction loss suffers from two problem : 1. Latent vector is not smooth and continuous, which might lead to memorising signals 2. Network is prone to outliers as mean squared error is used for reconstruction loss. We propose a regularised encoder decoder based architecture with KL divergence as regulariser for latent vector which solves the above two problem. The regulariser will enforce the network to minimise the distance between latent vector distribution and normal distribution, hence making latent vector smooth and continuous, at the same time as diverse as possible.```

### Directory structure :
1. kernels : Contains notebooks for various methods proposed in the network
2. models : Contains weights for the trained networks

Codes are implemented using `pytorch 1.0`.

### Kernels description :
1. ConvAutoEncoder : Implementation of standard 1D CNN autoencoder
2. LSTMAutoEncoder : Implementation of standard LSTM autoencoder
3. KLAutoEncoder : Implementation of proposed regualrised 1D CNN autoencoder
4. KLLSTMAutoEncoder : Implementation of proposed regualrised lstm autoencoder
