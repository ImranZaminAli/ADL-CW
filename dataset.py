import pandas as pd
import numpy as np
import torch

from torch.utils import data

class MagnaTagATune(data.Dataset):
    def __init__(self, dataset_path, samples_path):
        """
        Given the dataset path, create the MagnaTagATune dataset. Creates the
        variable self.dataset which is a list of 3-element tuples, each of the
        form (filename, samples, label):
            1) The filename which a given set of audio samples belongs to
	        2) The audio samples which relates to a 29.1 second music clip
               resampled to 12KHz. Each array is of shape: [10, 1, 34950].
               Where 10 represents the sub-clips, 1 is the channel dim and
               34950 are the number of samples in the sub-clip.
	        3) The multiclass label of the audio file of shape: [50].
        Args:
            dataset_path (str): Path to train_labels.pkl or val_labels.pkl
        """
        print(f"Loading data from {dataset_path}...")
        self.dataset = pd.read_pickle(dataset_path)
        self.samples_path = samples_path

    def __getitem__(self, index):
        """
        Given the index from the DataLoader, return the filename, spectrogram,
        and label
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            filename (str): the filename of the .wav file the spectrogram
                belongs to.
            samples (torch.FloatTensor): the audio samples of a 29.1
                second audio file.
            label (toch.FloatTensor): the class of the file/audio samples.
        """
        data = self.dataset.iloc[index]

        filename = data['file_path']
        samples = torch.from_numpy(np.load(f"{self.samples_path}/{filename}"))
        label = torch.FloatTensor(data['label'])
        samples = samples.view(10, -1).contiguous() # Create 10 subclips

        return filename, samples.unsqueeze(1), label

    def __len__(self):
        """
        Returns the length of the dataset (length of the list of 4-element
            tuples). __len()__ always needs to be defined so that the DataLoader
            can create the batches
        Returns:
            len(self.dataset) (int): the length of the list of 4-element tuples.
        """
        return self.dataset.shape[0]
