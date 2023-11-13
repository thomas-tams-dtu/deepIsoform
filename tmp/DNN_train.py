import IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm


import torch
from torch import nn
import numpy as np
from typing import *
from sklearn.decomposition import IncrementalPCA

class FeedForwardIsoform(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, output_shape:torch.Size) -> None:
        super(FeedForwardIsoform, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_features = np.prod(input_shape)
        self.output_features = np.prod(output_shape)

        self.FNN = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=512).double(),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512).double(),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.output_features).double()
        )

        for layer in self.FNN:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x = x.view(x.size(0), -1)
        x = self.FNN(x)

        return x

BATCH_SIZE = 64
LATENT_FEATURES = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
MODEL_NAME = f'PCA_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}'

# Here is an example of loading the Archs4 gene expression dataset and looping over it
# If you have about 12GB of memory, you can load the dataset to memory using the argument load_in_mem=True
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")

archs4_train_dataloader = DataLoader(archs4_train, batch_size=64, shuffle=True)

ipca = IncrementalPCA(n_components=32)

count = 0
for X in tqdm(archs4_train_dataloader):
    #if count > 50:
        #break
    count += 1
    ipca.partial_fit(X.numpy())


# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))


gtx_train_dataloader = DataLoader(gtex_train, batch_size=BATCH_SIZE, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=BATCH_SIZE, shuffle=True)
expr = next(iter(gtx_train_dataloader))

FNN = FeedForwardIsoform(input_shape = LATENT_FEATURES, 
                         output_shape = expr[1][0].size())


# Loss function
criterion = torch.nn.CrossEntropyLoss()

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(FNN.parameters(), lr=LEARNING_RATE)


print(FNN)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f">> Using device: {device}")

# move the model to the device
FNN = FNN.to(device)

# training..
NUM_EPOCHS = 10
epoch = 0
loss_over_time = []
while epoch < NUM_EPOCHS:
    epoch+= 1
    print("Epoch:", epoch)
    FNN.train()
    for X,y in tqdm(gtx_train_dataloader):
        #Encode X using VAE encoder to latent space
        X = ipca.transform(X)
        X = torch.from_numpy(X)
    
        X = X.to(device)
        y = y.to(device)
        #print(X.shape, y.shape)
        X = FNN.forward(X)
        
        loss = criterion(X, y).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


    with torch.no_grad():
        FNN.eval()
        X, y = next(iter(gtx_test_dataloader))
        
        X = ipca.transform(X)
        X = torch.from_numpy(X)
        X = X.to(device)
        y = y.to(device)
        print(y.shape)
        X = FNN.forward(X)
        loss = criterion(X, y).to(device)
        loss_over_time.append(loss)
        print(loss)

