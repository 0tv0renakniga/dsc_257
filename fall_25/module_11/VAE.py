# %% [markdown]
# # <font color='blue'>Variational autoencoder (VAE) for MNIST</font>
# 
# In this notebook, we explore Variational Autoencoders (VAEs). VAEs are powerful generative models that can learn to encode and decode complex data distributions, making them suitable for tasks like image generation, data denoising, and latent space manipulation. We'll apply them to the MNIST dataset of handwritten digits and demonstrate how to build and train a VAE from scratch.
# 
# The VAE is a probabilistic model that learns a compact representation of data while enabling the generation of new data points. It has two components: an <b>encoder</b> network that maps data points to a latent space and a <b>decoder</b> network that generates data points from the latent space representations. VAEs use probabilistic distributions to model the latent space, which allows for sampling and interpolation in this space.
# 
# Here's a high-level overview of what we'll cover:
# 
# 1. Introduction to VAEs: the core concepts behind VAEs, including the probabilistic framework and the objective function.
# 
# 2. Setting up the MNIST dataset: loading and preprocessing the MNIST dataset, preparing it for training our VAE.
# 
# 3. Building the VAE architecture: defining the encoder and decoder networks.
# 
# 4. Training the VAE: training the VAE on the MNIST dataset, by minimizing a particular loss function. 
# 
# 5. Sampling and generating new images.
# 
# 6. Visualizing the latent space.
# 
# Note: This notebook builds upon code from https://github.com/lyeoni/pytorch-mnist-VAE

# %% [markdown]
# ## <font color='blue'>Introduction to VAEs</font>
# 
# VAEs extend the basic autoencoder concept by introducing a probabilistic framework. Rather than mapping data directly to a fixed point in the latent space, VAEs model the data generation process probabilistically.
# 
# The core components of VAEs include:
# 
# 1. Encoder network: This part of the model maps an input data point to the parameters of a (Gaussian) distribution over the latent space.
# 
# 2. Reparameterization trick: To make the model differentiable, VAEs use a reparameterization trick, enabling them to sample from the latent distribution without breaking the backpropagation chain. 
# 
# 3. Decoder network: The decoder network takes a vector in the latent space and generates a point in the data (input) space.
# 
# 4. Variational loss: VAEs are trained to minimize a loss function composed of two terms: a reconstruction loss, ensuring the generated data resembles the input, and a regularization term encouraging the latent space to follow a particular distribution (usually a standard Gaussian).

# %% [markdown]
# ## <font color='blue'>Package setup and dataset</font>
# 
# In this notebook, we'll be working with the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits from 0 to 9.
# 
# In the provided code block, we begin by importing the necessary libraries, including `PyTorch` and `torchvision`, which will help us manage the dataset effortlessly. We also set the batch size `bs` to 100 for our data loader, which determines how many samples are processed in each training iteration.
# 
# Next, we use torchvision to download and preprocess the MNIST dataset. The dataset is automatically downloaded, transformed into PyTorch tensors, and loaded into memory. We shuffle the data by setting `shuffle=True` in the data loader.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid
from torchinfo import summary
from matplotlib import pyplot as plt

# %%
# Batch size for training
bs = 100
# Loader for training set
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
# For this notebook 'cpu' should be fine
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# Let's look at the first five images (data points) returned by the data loader.

# %%
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, (image, label) in enumerate(train_loader):
    if i >= 5:
        break
    ax = axes[i]
    ax.imshow(image[0, 0], cmap='gray')
    ax.set_title(f"Label: {label[0].item()}")
    ax.axis('off')
plt.show()

# %% [markdown]
# ## <font color='blue'>Defining the VAE</font>
# 
# In this section, we define the VAE architecture. There is an encoder network responsible for mapping input data to a latent space, a decoder network for generating data from latent space representations, and additional layers to facilitate the probabilistic framework of VAEs.
# 
# The VAE class is implemented as a PyTorch module and is initialized with two crucial parameters: `x_dim` and `z_dim`. These parameters represent the dimensions of the input data and the latent space, respectively.
# 
# Here's a breakdown of the components of the VAE class:
# 
# 1. Encoder network: This has two fully connected layers (`downsize_1` and `downsize_2`) that reduce the dimensionality of the input data. The output of these layers is then split into two branches, with one branch predicting the mean (`mu_fc`) and the other branch predicting the log variance (`log_var_fc`) of the distribution over the latent space.
# 
# 2. Sampling function: The sampling method generates samples from the latent space using the reparameterization trick. It takes the predicted mean and log variance, transforms the log variance into standard deviation, and samples random noise to create the latent code.
# 
# 3. Decoder network: The decoder network reverses the process by mapping the latent space back to the data space. It consists of three fully connected layers (`upsize_1`, `upsize_2`, and `upsize_3`) that progressively increase the dimensionality.
# 
# 4. Forward method: In the forward method, we connect the encoder, sampling, and decoder to create a complete forward pass. Given an input `x`, it first uses the encoder to compute the mean and log variance of a Gaussian distribution over the latent space. It then samples from the latent space and feeds this sample to the decoder to produce a reconstruction of the input data. The method also returns the mean and log variance for later use during training.
# 
# We'll proceed to train this VAE and generate new images in the upcoming sections.

# %%
class VAE(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(VAE, self).__init__()
        
        self.downsize_1 = nn.Linear(x_dim, 512)
        self.downsize_2 = nn.Linear(512, 256)
        self.mu_fc = nn.Linear(256, z_dim)
        self.log_var_fc = nn.Linear(256, z_dim)
        
        self.upsize_1 = nn.Linear(z_dim, 256)
        self.upsize_2 = nn.Linear(256, 512)
        self.upsize_3 = nn.Linear(512, x_dim)
        
    def encoder(self, x):
        h = nn.functional.relu(self.downsize_1(x))
        h = nn.functional.relu(self.downsize_2(h))
        return self.mu_fc(h), self.log_var_fc(h) 
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
        
    def decoder(self, z):
        h = nn.functional.relu(self.upsize_1(z))
        h = nn.functional.relu(self.upsize_2(h))
        return nn.functional.sigmoid(self.upsize_3(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# %% [markdown]
# ## <font color='blue'>Training the VAE</font>
# 
# The first step is to define the loss function.
# 
# ### <font color='blue'>The loss function</font>
# 
# The loss function for VAEs balances reconstruction quality with regularization of the latent space.
# 
# The `loss_function` defined in this code block plays a pivotal role in training our VAE. It has two components:
# 
# - Binary cross-entropy loss (BCE): measures the dissimilarity between the reconstructed data and the original input. It quantifies how well the VAE does at reproducing the input data. We use the `nn.BCELoss` function with the `reduction='sum'` argument to compute the BCE loss. It compares the reconstructed data (`recon_x`) and the input data (`x.view(-1, 784)`) element-wise.
# 
# - Kullback-Leibler divergence (KLD): regularizes the latent space. It encourages the distribution of the latent variables (encoded by `mu` and `logvar`) to be as close as possible to a standard Gaussian distribution. The formula used here is the Kullback-Leibler divergence, which quantifies the difference between two probability distributions. 
# 
# The total loss is the sum of the `BCE` and `KLD` terms. This combination of terms results in the Evidence Lower Bound (ELBO), which we aim to maximize during training. Maximizing the ELBO is equivalent to maximizing the likelihood of the data under the model while regularizing the latent space.
# 

# %%
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x.view(-1,784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# %% [markdown]
# In this section, we configure some crucial parameters for the VAE model and for training.
# 
# 
# - Latent dimension (`latent_dim`): dimensionality of the latent space in our VAE. In this example, we set it to 8, but you can adjust it depending on the complexity of the dataset and the desired level of data compression.
# 
# - Model initialization: We create an instance of our VAE model by calling `VAE(x_dim=784, z_dim=latent_dim)`. This step initializes the VAE architecture. The `x_dim` is set to `784`, corresponding to the dimensionality of the input data (28x28 images flattened to 784 pixels).
# 
# - Optimizer initialization: We train with the Adam optimizer, a popular choice for deep learning models. The `optim.Adam` function is used to initialize the optimizer, and it is provided with the model's parameters. The optimizer will be responsible for adjusting the model's weights during training to minimize the defined loss function.

# %%
# Key parameter: dimension of latent space
latent_dim = 8

model = VAE(x_dim=784,z_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters())

# %% [markdown]
# ### <font color='blue'>The training procedure</font>
# 
# This next cell trains the VAE for a specified number of epochs.
# 
# You don't need to run the cell at this point since we have already done it for two choices of `latent_dim` (dimension of `z`): 2 and 8. These models are stored in the files `vae_mnist_z_2.pth` and `vae_mnist_z_8.pth`.
# 

# %%
#
# NO NEED TO RUN THIS CELL UNLESS YOU WANT TO TRAIN YOUR OWN MODEL
#
num_epochs = 50 # number of epochs of training
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        recon_data, mu, logvar = model(data)
        loss = loss_function(recon_data, data, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Epoch {} Loss : {:.4f}".format(epoch, epoch_loss / len(train_dataset)))
# Save learned model
torch.save(model.state_dict(), 'vae_mnist_z_{}.pth'.format(latent_dim))

# %% [markdown]
# ## <font color='blue'>Loading the pre-trained model</font>
# 
# With a pretrained VAE model loaded, you can now perform various tasks, such as generating new images, exploring the latent space, or evaluating the model's performance on different datasets.
# 
# First, let's load a model (for a specified `latent_dim`) and print a summary.

# %%
latent_dim = 8
model_test = VAE(x_dim=784,z_dim=latent_dim).to(device)
model_test.load_state_dict(torch.load('vae_mnist_z_{}.pth'.format(latent_dim),map_location=torch.device('cpu')))
summary(model_test, input_size=(100, 1, 28, 28))

# %% [markdown]
# ## <font color='blue'>Generating images from the VAE</font>
# 
# In this section, we use the VAE to generate new MNIST-like samples.
# 
# We pick 100 random vectors in the latent space (of dimension `latent_dim`), each from a standard Gaussian distribution. This random sampling allows us to observe the diversity of images that the VAE can generate.
# 
# For each of these random latent vectors, we use our decoder (`model_test.decoder`) to generate a corresponding image in the usual (784-dimensional) MNIST space. These images are saved in the folder `samples` for later inspection.

# %%
with torch.no_grad():
    random_sample = torch.randn(100,latent_dim)
    generated_image = model_test.decoder(random_sample.to(device))
    save_image(generated_image.view(100,1,28,28),'./samples/random_sample_z_{}.png'.format(latent_dim), nrow=10)

# %% [markdown]
# Now let's view the random samples

# %%
img = plt.imread('./samples/random_sample_z_{}.png'.format(latent_dim))
plt.title('Randomly generated examples (latent dim {})'.format(latent_dim))
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
plt.imshow(img)
plt.show()

# %% [markdown]
# ## <font color='blue'>Further exploration of the latent space</font>
# 
# Next, we explore the latent space a bit more closely.
# 
# This next cell generates 20 large image files, each containing a grid of 256 images. Each of the large files is produced in the following way:
# - We pick two random points in the latent space (of dimension `latent_dim`). Think of one as the starting point and the other as the ending point.
# - We move in the latent space from the starting point to the ending point. Specifically, we generate 256 vectors spaced evenly on the line segment joining the starting and ending points.
# - For each of these 256 latent vectors, we use our decoder (`model_test.decoder`) to generate a corresponding image in the usual 784-dimensional MNIST space.
# - We thus get a sequence of 256 images in MNIST space and we save them in a 16-by-16 grid. 
# 
# This process is repeated 20 times, and the resulting large images are stored in the `.\samples` folder.

# %%
num_steps = 256
with torch.no_grad():
    for idx in range(20):
        # Pick a random start point and end point in the latent space
        start_point = torch.randn(1, latent_dim) 
        end_point = torch.randn(1, latent_dim)

        # Now look at num_steps points spaced evenly on the line between
        # the starting and ending points and generate an image from each 
        # of these latent vectors. Save the resulting chain of images.
        interpolated_points = torch.stack([start_point + (i / (num_steps - 1)) * (end_point - start_point) for i in range(num_steps)])
        generated_image = model_test.decoder(interpolated_points.to(device))
        save_image(generated_image.view(num_steps,1,28,28),'./samples/latent_exploration_z_{}_{}.png'.format(latent_dim,idx), nrow=int(num_steps**0.5))

# %% [markdown]
# Now let's take a look at the first of these 20 images.

# %%
img = plt.imread('./samples/latent_exploration_z_{}_1.png'.format(latent_dim))
plt.title('Latent space exploration (latent dim {})'.format(latent_dim))
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
plt.imshow(img)
plt.show()


