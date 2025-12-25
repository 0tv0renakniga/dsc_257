# %% [markdown]
# ## q2
# *Write a procedure that draws 10000 points $X_1, \ldots, X_{10000}$ from the spherical Gaussian $N(0, I_d)$ and plots a histogram of their (squared) lengths $\|X_i\|^2$.* 
# - Do this for $d=10$. What seems to be the typical (squared) length? 
# - Repeat for $d=100$. Is the histogram more concentrated than in (a), or is it less concentrated? 

# %%
import numpy as np
import matplotlib.pyplot as plt

# Set dimension and number of samples
d = 10
num_samples = 10000
np.random.seed(42) # for reproducibility

# Draw 10000 points from N(0, I_d)
# samples will have shape (num_samples, d)
samples = np.random.normal(0, 1, size=(num_samples, d))

# Calculate the squared L2 norm (||X||^2) for each point
# np.sum(samples**2, axis=1) gives a vector of 10000 squared lengths
squared_lengths_d10 = np.sum(samples**2, axis=1)

# Calculate sample mean
sample_mean_d10 = np.mean(squared_lengths_d10)
print(f"Dimension (d): {d}")
print(f"Sample Mean of ||X||^2: {sample_mean_d10:.4f}")

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(squared_lengths_d10, bins=50, density=True, alpha=0.7, 
         label=f'Histogram of {num_samples} samples')
plt.axvline(d, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Mean E[||X||^2] = d = {d}')
plt.axvline(sample_mean_d10, color='blue', linestyle=':', linewidth=2, 
            label=f'Sample Mean = {sample_mean_d10:.4f}')

plt.title('Histogram of Squared Lengths for $N(0, I_{10})$', fontsize=14)
plt.xlabel('Squared Length $||X||^2$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('histogram_d10.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def plot_squared_lengths(d, n_samples=10000, filename=None):
    # 1. Generate n_samples points from N(0, I_d)
    # Shape: (n_samples, d)
    X = np.random.randn(n_samples, d)
    
    # 2. Compute squared lengths: sum(X_ij^2) along axis 1
    squared_lengths = np.sum(X**2, axis=1)
    
    # Calculate statistics for analysis
    mean_sq_len = np.mean(squared_lengths)
    std_sq_len = np.std(squared_lengths)
    print(f"d={d}: Mean = {mean_sq_len:.4f}, Std Dev = {std_sq_len:.4f}")
    print(f"d={d}: Std/Mean Ratio = {std_sq_len / mean_sq_len:.4f}")

    # 3. Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(squared_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(d, color='red', linestyle='dashed', linewidth=2, label=f'Theoretical Mean ({d})')
    plt.title(f'Histogram of Squared Lengths $\|X\|^2$ for $d={d}$')
    plt.xlabel('Squared Length')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# Execute for d=10
plot_squared_lengths(d=10, filename='histogram_d10.png')

# %%
import numpy as np
import matplotlib.pyplot as plt

# set dimension and number of samples
d = 10
num_samples = 10000
np.random.seed(42) 

# get 10000 points from N(0, I_d)
samples = np.random.normal(0, 1, size=(num_samples, d))

# calc the squared L2 norm for each point
squared_lengths_d10 = np.sum(samples**2, axis=1)

# calculate mean
sample_mean_d10 = np.mean(squared_lengths_d10)
print(f"Dimension (d): {d}")
print(f"Sample Mean of ||X||^2: {sample_mean_d10:.4f}")

# plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(squared_lengths_d10, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True,label=f'Histogram of {num_samples} samples')
plt.axvline(d, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Mean (d = {d}): $E[||X||^2]$')
plt.axvline(sample_mean_d10, color='blue', linestyle=':', linewidth=2, 
            label=f'Sample Mean = {sample_mean_d10:.4f}')
# set plot args and save
plt.title('Histogram of Squared Lengths for $N(0, I_{10})$', fontsize=14)
plt.xlabel('Squared Length $||X||^2$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('histogram_d10.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# set dimension and number of samples
d = 100
num_samples = 10000
np.random.seed(42) 

# get 10000 points from N(0, I_d)
samples = np.random.normal(0, 1, size=(num_samples, d))

# calc the squared L2 norm for each point
squared_lengths_d100 = np.sum(samples**2, axis=1)

# calculate mean
sample_mean_d100 = np.mean(squared_lengths_d100)
print(f"Dimension (d): {d}")
print(f"Sample Mean of ||X||^2: {sample_mean_d100:.4f}")

# plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(squared_lengths_d100, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True,label=f'Histogram of {num_samples} samples')
plt.axvline(d, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical Mean (d = {d}): $E[||X||^2]$')
plt.axvline(sample_mean_d100, color='blue', linestyle=':', linewidth=2, 
            label=f'Sample Mean = {sample_mean_d100:.4f}')
# set plot args and save
plt.title('Histogram of Squared Lengths for $N(0, I_{100})$', fontsize=14)
plt.xlabel('Squared Length $||X||^2$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('histogram_d100.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Helper Functions (from 5b) ---
def load_data(filename):
    """Loads the CSV data, assuming no header."""
    data = pd.read_csv(filename, header=None)
    data.columns = ['latitude', 'longitude', 'temperature']
    X = data[['latitude', 'longitude']].values
    y = data['temperature'].values
    return X, y

def kernel(X1, X2, sigma=1.5):
    """Computes the exponential kernel: k(x, x') = exp(-||x - x'|| / sigma)"""
    dists = cdist(X1, X2, metric='euclidean')
    return np.exp(-dists / sigma)

# --- 1. Load Data and Setup Parameters ---
X_tr, y_tr = load_data('gptrain.csv')
SIGMA = 1.5
MEAN_VAL = 20.0
JITTER = 1e-6

# --- 2. Pre-compute for Posterior (from 5b) ---
m_tr = np.full(y_tr.shape, MEAN_VAL)
K_tr = kernel(X_tr, X_tr, sigma=SIGMA)
K_tr_stable = K_tr + np.eye(K_tr.shape[0]) * JITTER
alpha = np.linalg.solve(K_tr_stable, y_tr - m_tr)

# --- 3. Create Rich Grid (for 5c) ---
# We use 80x80 = 6400 points
GRID_SIZE = 80 
lat_min, lat_max = X_tr[:, 0].min(), X_tr[:, 0].max()
lon_min, lon_max = X_tr[:, 1].min(), X_tr[:, 1].max()

lat_grid = np.linspace(lat_min, lat_max, GRID_SIZE)
lon_grid = np.linspace(lon_min, lon_max, GRID_SIZE)
# np.meshgrid creates [LON, LAT] for (x, y) plotting convention
LON, LAT = np.meshgrid(lon_grid, lat_grid)

# Stack into (n_grid_points, 2) array
X_grid = np.column_stack([LAT.ravel(), LON.ravel()])
print(f"Created grid with {X_grid.shape[0]} points.")

# --- 4. Compute Posterior Mean on Grid ---
m_grid = np.full(X_grid.shape[0], MEAN_VAL)
# K_grid_tr = K(X_grid, X_tr) -> (6400, 93)
K_grid_tr = kernel(X_grid, X_tr, sigma=SIGMA)

# mu_grid = m_grid + K_grid_tr @ alpha
mu_grid = m_grid + K_grid_tr @ alpha
mu_grid_reshaped = mu_grid.reshape(LAT.shape)

# --- 5. Compute Posterior Standard Deviation on Grid ---
# We need diag(K_grid - K_grid_tr @ K_tr_inv @ K_tr_grid)
# diag(K_grid) is 1.0 everywhere since k(x,x) = 1
K_diag = np.ones(X_grid.shape[0]) 

# Use Cholesky decomposition for stability
# L @ L.T = K_tr_stable
L = np.linalg.cholesky(K_tr_stable) 
# Solve L @ V = K_grid_tr.T for V
# V = L_inv @ K_grid_tr.T
V = np.linalg.solve(L, K_grid_tr.T) # Shape (93, 6400)

# diag(K_grid_tr @ K_tr_inv @ K_tr_grid) = diag(V.T @ V) = sum(V**2, axis=0)
diag_term = np.sum(V**2, axis=0) # Shape (6400,)

# Posterior variance
var_grid = K_diag - diag_term
# Clip at 0 for numerical stability before sqrt
std_grid = np.sqrt(np.maximum(var_grid, 0)) 
std_grid_reshaped = std_grid.reshape(LAT.shape)

# --- 6. Plot Results ---
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
plot_extent = [lon_min, lon_max, lat_min, lat_max]

# Plot 1: Posterior Mean
im1 = ax[0].imshow(mu_grid_reshaped, extent=plot_extent, 
                   origin='lower', aspect='auto', cmap='hot')
ax[0].scatter(X_tr[:, 1], X_tr[:, 0], c='blue', marker='x', label='Training Stations')
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')
ax[0].set_title('GP Posterior Mean Temperature')
ax[0].legend()
fig.colorbar(im1, ax=ax[0], label='Temperature (°C)')

# Plot 2: Posterior Standard Deviation
im2 = ax[1].imshow(std_grid_reshaped, extent=plot_extent, 
                   origin='lower', aspect='auto', cmap='viridis')
ax[1].scatter(X_tr[:, 1], X_tr[:, 0], c='red', marker='x', label='Training Stations')
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_title('GP Posterior Standard Deviation')
ax[1].legend()
fig.colorbar(im2, ax=ax[1], label='Std. Deviation (°C)')

plt.tight_layout()
# Save the figure to be included in the LaTeX document
plt.savefig('gp_predictions.png', dpi=150, bbox_inches='tight')
print("Saved plots to gp_predictions.png")

# %% [markdown]
# ## <font color='blue'>Text generation using a Char-RNN model</font>
# 
# We're going to train a Recurrent Neural Network (RNN) to understand and generate text character by character. To do this, we'll provide the RNN with a large piece of text and ask it to learn the likelihood of the next character based on the sequence of previous characters.
# 
# Let's break it down with a simple example: Imagine our vocabulary consists of just four letters, "helo," and our training sequence is "hello." In this case, we have four separate training examples:
# 
# - The RNN should learn that when it sees "h", the next character "e" is likely.
# - When it encounters "he", it should expect "l" to come next.
# - Similarly, when it has "hel" as input, it should predict "l".
# - Finally, after "hell", it should anticipate "o".
# 
# To make this happen, we'll represent each character as a vector using a technique called 1-of-k encoding, where each character is uniquely identified by a specific position in the vector. We'll then feed these character vectors into the RNN one at a time using a step function. The RNN will produce a sequence of output vectors, each with four dimensions, corresponding to the likelihood of the next character in the sequence.
# 
# In essence, we're training the RNN to understand and generate text character by character, and it will predict the next character based on the context of the preceding characters.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
import numpy as np

# %% [markdown]
# ### <font color='blue'>Some pre-processing</font>
# 
# We will train our model using a text file of Shakespeare's plays. 
# 
# The first step is create a mapping from characters to integers, so as to represent each string as a list of integers. This is essential since we can only pass in numbers to our model, not strings or characters. Using this mapping, we now have our corpus of text mapped into a list of numbers.

# %%
# Create a character-to-index and index-to-character mapping
chars = np.load('chars.npy')
# np.save('chars.npy', chars)
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# %% [markdown]
# Let's examine the mapping between integers and characters

# %% [markdown]
# <font color='magenta'>(a) By looking at the dictionary `char_to_index`, answer the following questions:</font>
# * <font color='magenta'>How many characters are we considering?</font>
# * <font color='magenta'>What is the code for A?</font>

# %% [markdown]
# Now let's read in Shakespeare's plays and convert the text to integers.

# %%
text = open('shakespeare_plays.txt', 'r').read()

# Convert the text to a numerical sequence
# text_as_int = [char_to_index[char] for char in text]

data = list(text)
for i, ch in enumerate(data):
    data[i] = char_to_index[ch]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data tensor on device
data = torch.tensor(data).to(device)
data = torch.unsqueeze(data, dim=1)

# %% [markdown]
# <font color='magenta'>(b) What is the length of the corpus in characters?</font>

# %% [markdown]
# ### <font color='blue'>Defining our model</font>
# 
# ##### Initialization:
# 
#   The `__init__` method initializes the RNN model with the following parameters:
#   - input_size: The size of the character vocabulary. This indicates the number of unique characters that the model can work with.
#   - output_size: The size of the output vocabulary. It's typically set to the same value as input_size for character generation tasks.
#   - hidden_size: The number of hidden units in the LSTM (Long Short-Term Memory) layer.
#   - num_layers: The number of LSTM layers stacked on top of each other.
# 
# ##### Embedding Layer:
# 
#   Inside the `__init__` method, an `nn.Embedding` layer is created. This layer is used to convert character indices (input) into dense vectors of fixed size. 
# 
# ##### LSTM Layer:
# 
# The `nn.LSTM layer` is defined with the specified `input_size`, `hidden_size`, and `num_layers`. This LSTM layer will process the embedded character sequence to capture dependencies and patterns within the sequence.
# 
# ##### Decoder Layer:
# 
# After the LSTM layer, there is a linear (fully connected) layer defined as `nn.Linear`, which takes the output from the LSTM layer and maps it to the desired output size. 
# 
# ##### Forward Pass:
# 
# The forward method is where the actual computation occurs. It takes an input sequence (`input_seq`) and a hidden state (`hidden_state`) as input arguments.
# 
# First, the input sequence is passed through the embedding layer to convert the character indices into dense embeddings.
# 
# Then, these embeddings are fed into the LSTM layer, which processes the sequence. The LSTM layer produces an output sequence (output) and an updated hidden state.
# 
# Finally, the output from the LSTM is passed through the linear decoder layer to generate the predictions for the next characters in the sequence.
# 
# The forward method returns the output sequence and the updated hidden state.
# 
# Note that the `self.rnn` is actually an LSTM. This is used since LSTM's are known to outperform RNNs in most language tasks. We can very well replace this with an RNN, but would expect the model not to perform that well.

# %%
# Define the Char-RNN Model
class CharRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())

# %% [markdown]
# ### <font color='blue'>Defining a dataset class</font>
# 
# In this part of the tutorial, we'll create a custom PyTorch dataset called `TextDataset`. This dataset is designed for training character-level text generation models like CharRNN. The dataset allows you to prepare your text data for training by converting characters to integer indices and creating input-target pairs for the model.
# 
# 
# 
# ##### Initialization:
# 
# Accepts three parameters: `text`, `seq_length`, and `char_to_index`.
# 
# - `text`: The input text data you want to train the model on.
# - `seq_length`: The length of sequences to be used during training (e.g., 50 characters per sequence).
# - `char_to_index`: A dictionary mapping characters to integer indices.
# 
# ##### Conversion of Text to Integers:
# 
# Inside the constructor, the input text is converted into an integer representation by mapping characters to their corresponding integer indices using the `char_to_index` dictionary.
# 
# ##### `__len__`:
# 
# Defines the length of the dataset. You can specify a fixed length (e.g., 10,000) for your dataset, but this can be adjusted based on your dataset size. What you can also do is simply set length as `len(text) - self.seq_length`. This would result in a much larger set of samples and you wouldn't need to randomly sample an index (as described next).
# 
# ##### `__getitem__`:
# 
# 
# Retrieves individual training examples from the dataset.
# 
# - Randomly selects a starting index within the range `[0, len(text) - seq_length)` for each training example.
# - Creates an input sequence (`input_seq`) containing characters from the selected `index` to `index + seq_length`.
# - Creates a target sequence (`target_seq`) containing characters from `index + 1` to `index + seq_length + 1`.
# - Returns a tuple with `input_seq` and `target_seq`.

# %%
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, seq_length, char_to_index):
        self.seq_length = seq_length
        self.char_to_index = char_to_index
        self.text_as_int = [char_to_index[char] for char in text]

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.text_as_int) - self.seq_length)
        input_seq = torch.tensor(self.text_as_int[idx:idx + self.seq_length])
        target_seq = torch.tensor(self.text_as_int[idx + 1:idx + self.seq_length + 1])
        return input_seq, target_seq

# Create the dataset
seq_length = 100
text_dataset = TextDataset(text, seq_length, char_to_index)

# Create a data loader
batch_size = 2048
data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

# %%
# Define the training loop

input_size = len(chars)
output_size = len(chars)
hidden_size = 512
num_layers = 3

model = CharRNN(input_size, output_size, hidden_size, num_layers)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 15

# Training device
model = model.to(device)


# %% [markdown]
# We can now train our model using the following code. For ease of use, a pre-trained model has been provided since training the model can be a long process especially if you don't have GPUs set up on your local machine.

# %%
## NO NEED TO RUN THIS CELL

# for i_epoch in range(1, num_epochs+1):
    
#     n = 0
#     running_loss = 0
#     hidden_state = None
    
#     for i_data,(input_seq, target_seq) in enumerate(data_loader):
#         print(i_data)
#         # forward pass
#         input_seq = input_seq.to(device)
#         target_seq = target_seq.to(device)
#         output, hidden_state = model(input_seq, hidden_state)
#         print(output.shape,target_seq.shape)
#         # compute loss
#         loss = criterion(output.view(-1,output_size), target_seq.view(-1))
#         running_loss += loss.item()
        
#         # compute gradients and take optimizer step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         n +=1
        
         
#     # print loss and save weights after every epoch
#     print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
#     torch.save(model.state_dict(), './model_{}.pth'.format(i_epoch))




# %% [markdown]
# Let's load the pretrained weights

# %%
model.load_state_dict(torch.load('./CharRNN_shakespeare.pth',map_location=torch.device('cpu')))
model = model.cpu()
model.eval()

# %% [markdown]
# <font color='magenta'>(c) When you ran the cell above, you should have gotten an input and output size of 66. Where is this number coming from?</font>

# %% [markdown]
# Time to generate some Shakespeare!

# %%
input_seq = data[25:26].cpu()
hidden_state = None
o_len = 0
output_len = 2000
while o_len < output_len:
    # forward pass
    output, hidden_state = model(input_seq, hidden_state)
    # construct categorical distribution and sample a character
    output = torch.nn.functional.softmax(torch.squeeze(output), dim=0)
    dist = torch.distributions.Categorical(output)
    index = dist.sample()
    # index = torch.argmax(output)
    # print the sampled character
    print(index_to_char[index.item()], end='')
    
    # next input is current output
    input_seq[0][0] = index.item()
    o_len += 1


