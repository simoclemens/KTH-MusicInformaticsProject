import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# AUDIO DATASET
# built in order to work directly with features loaded from a json in the format used by feature extraction
class AudioTrackDataset(Dataset):
    def __init__(self, mode, annotations_path, name, custom_duration=True, **kwargs):

        self.mode = mode  # 'train', 'val' or 'test'
        self.annotations_path = annotations_path
        self.name = name
        self.custom_duration = custom_duration

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            file_name = "train_" + name
        else:
            file_name = "test_" + name

        if self.custom_duration:
            file_name += "_with_duration"

        file_name += ".json"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.track_list = pd.read_json(os.path.join(self.annotations_path, file_name))
        print(f"Dataloader for {name}-{self.mode} with {len(self.track_list)} samples generated")

        # generate a list of dictionaries with uid and label for each action
        self.track_list = [{'index': item[1]['index'],
                            'bpm': item[1]['verb_class'],
                            'key': item[1]['key'],
                            'features': item[1]['features']
                            }
                           for item in self.track_list.iterrows()]

    def __getitem__(self, index):
        # record is a row of the pkl file containing one sample/action
        track = self.track_list[index]
        features = track['features']
        label = track['key']
        if self.custom_duration:
            output = {'features': np.array(features), 'label': label}
        else:
            output = {'features': np.array(features[0]), 'label': label}
        return output

    def __len__(self):
        return len(self.track_list)
