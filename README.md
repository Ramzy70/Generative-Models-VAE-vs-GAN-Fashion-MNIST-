# Generative Models: VAE vs GAN (Fashion-MNIST)

This repository contains an implementation and comparative analysis of two fundamental generative models: **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**. Both models are trained on the **Fashion-MNIST** dataset to evaluate their trade-offs in image quality, training stability, and latent space structure.

## 1. Variational Autoencoder (VAE)
The VAE is implemented as a **Convolutional VAE** to enhance feature extraction from the fashion item images.

### Architecture
* **Encoder:** Consists of two `Conv2d` layers with stride 2 and ReLU activations. It maps the $28\times28$ input image to a probabilistic latent space.
* **Latent Space:** Dimensionality of **20**. It utilizes the **Reparameterization Trick** ($z = \mu + \sigma \cdot \epsilon$) to allow backpropagation through stochastic nodes.
* **Decoder:** Utilizes `ConvTranspose2d` layers to upsample the latent vector back to the original dimensions, using a Sigmoid activation for the final output.



### Training & Loss
The model was trained for **20 epochs** using the **ELBO (Evidence Lower Bound)** loss function:
1.  **Reconstruction Loss:** Binary Cross-Entropy (BCE) measuring pixel-wise similarity.
2.  **KL Divergence:** Regularizes the latent space toward a Standard Normal Distribution $N(0,1)$.

**Final Performance Metrics:**
* **Total Loss:** ~240.7
* **Reconstruction Loss:** 218.36
* **KL Divergence:** 17.81

---

## 2. Generative Adversarial Network (GAN)
A **Deep Convolutional GAN (DCGAN)** architecture was chosen to generate sharp synthetic images.

### Architecture
* **Generator:** Uses `ConvTranspose2d` layers with **Batch Normalization** and ReLU activations to transform a random noise vector into a synthetic image.
* **Discriminator:** A convolutional binary classifier that distinguishes between real and synthetic images, utilizing LeakyReLU for better gradient flow.



### Training Stability
To stabilize training and prevent vanishing gradients, **Label Smoothing** (0.9 for real labels) was implemented to handle the inherent oscillations of the adversarial loss.

---

## 3. Comparison & Results

| Feature | Variational Autoencoder (VAE) | Generative Adversarial Network (GAN) |
| :--- | :--- | :--- |
| **Image Sharpness** | Tends to be blurry due to pixel-wise averaging. | Produces sharp details as the loss penalizes blurriness. |
| **Latent Space** | Continuous; allows for smooth interpolation. | Often discontinuous; interpolation can yield artifacts. |
| **Training Stability** | High stability; loss converges monotonically. | Low stability; loss oscillates (requires tuning). |

### Summary of Findings
* **VAE:** A robust tool for structured representation learning and encoding/manipulation of images.
* **GAN:** Superior for generating visually convincing, high-frequency details, though it lacks an explicit inference mechanism.

## Future Work
Future iterations could explore **VQ-VAE** or **WGAN-GP** to combine the structural benefits of VAEs with the visual fidelity of GANs.

---

## Setup and Requirements
* PyTorch
* Torchvision
* Matplotlib
* Numpy
* Pandas

To train the models and view the comparative plots, run the provided Jupyter Notebook: `test2.ipynb`.
