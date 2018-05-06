# VAE-example

VAE example from this [tutorial](https://blog.keras.io/building-autoencoders-in-keras.html)

For running the code just run:
```
python train.py
```

And the VAE will be trained and some interpolations from the encoded latent space will be displayed.

How does a variational autoencoder work?

First, an encoder network turns the input samples x into two parameters in a latent space, which we will note z_mean and z_log_sigma. Then, we randomly sample similar points z from the latent normal distribution that is assumed to generate the data, via z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor. Finally, a decoder network maps these latent space points back to the original input data.

The parameters of the model are trained via two loss functions: a reconstruction loss forcing the decoded samples to match the initial inputs (just like in our previous autoencoders), and the KL divergence between the learned latent distribution and the prior distribution, acting as a regularization term. You could actually get rid of this latter term entirely, although it does help in learning well-formed latent spaces and reducing overfitting to the training data.

Beta-VAE are a newer type of VAEs which tries to learn independent encoded features (of the Z vector, the latent normal distribution) and which add a Beta parameter to the loss (multiplying the KL divergence) and has to be set greater or equal to 1.
