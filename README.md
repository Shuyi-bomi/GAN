# GAN
Generative Adversarial Networks

In GANs, G could be regarded as a team of counterfeiters, trying to make counterfeit currency without being detected. D could be regarded as police, trying to identify counterfeit currency and give corresponding probability being real for each currency. We train them alternatively by Backpropagation and min-max game. Both G and D are multilayer perceptrons originally, whereas DCGAN arises afterwards which implements convolutional neural network (CNN) as G and D. As expected, generative samples are similar to real samples as well as the discriminative model converges to 0.5.

**data**: The dataset we used is planet data-mat format with dimension 3909 ∗ 4734. 

We intend to apply GAN and DCGAN to our data and see how it goes. In order to apply CNN as G and D, which is required in DCGAN, we firstly pad our data and transform to square image(69*69*1). Also, we normalize the data into range [0,1] using Minmax normalization.

But things don't go well. We find loss exploded and will never converge. Because when I apply GANs&DCGAN(design MLP/CNN) to represent G and D separately, I find that it would be easy for D to distinguish fake data/images which G produces from all inputs. So this will lead to training difficulties for G since it doesn’t know how to improve. At this point, D couldn’t converge to 1/2 as well. Like the following figure obtained from **Tensorboard** visualization tool of loss(DCGAN), we could see that both G and D exploded. Result obtained from GAN is similar.

<p align="middle">
  <img src="https://github.com/Shuyi-bomi/GAN/blob/main/result/D-loss(CNNs).png" width="410" />
  <img src="https://github.com/Shuyi-bomi/GAN/blob/main/result/G-loss(CNNs).png" width="200" /> 
</p>

#WGAN
In exploring the unstable behavior of GAN training, researchers demonstrate that Jensen-Shannon divergence between Pr and Pg is unable to provide useful gradient information to perform Gradient descent when Pr and Pg are disjoint[2, 1]. Then, Wasserstein distance is proposed and combined with Kantorovich-Rubinstein duality to generate new loss function. It’s continuous and differentiable almost everywhere under mild assumptions. To satisfy the mild assumptions, we need to clip the weight within [-c, c].

#WGAN-GP
However, WGAN still exist problem like training instability. Finally researcher finds out gradients may vanish quickly in original weight clipping. Finally, WGAN-GP apply an alternative way in weight clipping by implementing a k-Lipshitz constraint via gradient penalty.

We tried all these methods and it turned out WGAN-GP with MLP works best. We provide its loss figure obtained from Tensorboard visualization tool for G and D:
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/GAN/blob/main/result/D_lossmlp.png" width="200" />
  <img src="https://github.com/Shuyi-bomi/GAN/blob/main/result/G_lossmlp.png" width="200" /> 
</p>

So we could see for D it converged to 0.5, which testifies trained G and D work great since 0.5 is the convergence result for D in theoretic proof.

After we obtained the desired network, we then generated/simulated planet images from 'generate.py'. For the following figure, one image contains 16 data(16 ∗ 4761) simulated and we reshape each to 69 ∗ 69 dimension to visualize it. Every 1000 trainings we generated one image. And this is the one that we train 1000000 times.
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/GAN/blob/main/result/MLPs.png" width="200" />
</p>

The picture showed general characteristic for planet data, central value is relatively big and spherical is small.
