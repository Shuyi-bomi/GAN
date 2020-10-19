# GAN
Generative Adversarial Networks

In GANs, G could be regarded as a team of counterfeiters, trying to make counterfeit currency without being detected. D could be regarded as police, trying to identify counterfeit currency and give corresponding probability being real for each currency. We train them alternatively by Backpropagation and min-max game. Both G and D are multilayer perceptrons originally, whereas DCGAN arises afterwards which implements convolutional neural network (CNN) as G and D. As expected, generative samples are similar to real samples as well as the discriminative model converges to 0.5.

**data**: The dataset we used is planet data-mat format with dimension 3909 ∗ 4734. 

We intend to apply GAN and DCGAN to our data and see how it goes. In order to apply CNN as G and D, which is required in DCGAN, we firstly pad our data and transform to square image(69*69*1). Also, we normalize the data into range [0,1] using Minmax normalization.

But things don't go well. We find loss exploded and will never converge. Because when I apply GANs&DCGAN(design MLP/CNN) to represent G and D separately, I find that it would be easy for D to distinguish fake data/images which G produces from all inputs. So this will lead to training difficulties for G since it doesn’t know how to improve. At this point, D couldn’t converge to 1/2 as well. 
