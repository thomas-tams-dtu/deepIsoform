import pandas as pd
from torch.utils.data import Dataset

class VaeDataset(Dataset):
    def __init__(self, csv):
        # Read in the data
        self.df = pd.read_csv(path, sep="\t")
        # Extract the sample names
        self.sampleid = self.df["sample_id"].values
        # Extrac the values
        self.data = self.df.iloc[:,1:].values
    def __getitem__(self, idx):
        output = {}
        # Get the correct indices of data
        output["sample_id"] = self.sampleid[idx]
        output["data"] = self.data[idx]
        return output
        
    def __len__(self):
        # return the length of the data
        return len(self.data)