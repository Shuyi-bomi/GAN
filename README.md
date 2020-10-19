# GAN
Generative Adversarial Networks

In GANs, G could be regarded as a team of counterfeiters, trying to make counterfeit currency without being detected. D could be regarded as police, trying to identify counterfeit currency and give corresponding probability being real for each currency. We train them alternatively by Backpropagation and min-max game. Both G and D are multilayer perceptrons originally, whereas DCGAN arises afterwards which implements convolutional neural network (CNN) as G and D. 

**data**: The dataset we used is planet data-mat format with dimension 3909 ∗ 4734. 

We intend to apply GAN and DCGAN to our data and see how it goes. In order to apply CNN as G and D, which is required in DCGAN, we firstly pad our data and transform to square image(69*69*1). Also, we normalize the data into range [0,1] using Minmax normalization.
