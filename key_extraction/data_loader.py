import os
import json
from torch.utils.data import Dataset, DataLoader
import librosa


class AudioDataset(Dataset):
    def __init__(self, audio_folder, analysis_folder, transform=None, time_window=60):
        self.audio_folder = audio_folder
        audio_file_list = os.listdir(audio_folder)
        self.audio_file_list = [file_name.split(".")[0] for file_name in audio_file_list]
        self.analysis_folder = analysis_folder
        self.transform = transform
        self.time_window = time_window  # now using 60 seconds as default

        # key labels
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        self.key_mapping = {keys[i]: i for i in range(24)}

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, idx):
        audio_file_name = self.audio_file_list[idx]
        audio_file_path = os.path.join(self.audio_folder, audio_file_name)
        audio_features = self.pair_audio_with_features(audio_file_name)

        audio_data, sr = librosa.load(audio_file_path, sr=22050)
        frame_length = int(self.time_window * sr)
        num_frames = len(audio_data) // frame_length
        audio_data = audio_data[:num_frames * frame_length]  # only keep the first num_frames * frame_length samples
        chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)
        key, bpm = audio_features[audio_file_name]
        key_label = self.key_to_label(key)

        # also returing bpm?
        return chromagram, key_label

    # pair each file with its annotations
    def pair_audio_with_features(self, file_name):
        analysis_full_path = os.path.join(self.analysis_folder, file_name) + "_analysis.json"
        key = ""
        bpm = -1
        with open(analysis_full_path) as f:
            annotations = json.load(f)
            key = annotations["tonality"]
            bpm = annotations["tempo"]
            f.close()

        return {file_name: [key, bpm]}

    def key_to_label(self, key):
        return self.key_mapping.get(key, -1)


class AudioDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=collate_fn)


class FeatureDataset(Dataset):
    def __init__(self, audio_folder, analysis_folder, transform=None, time_window=60):
        self.audio_folder = audio_folder
        audio_file_list = os.listdir(audio_folder)
        self.audio_file_list = [file_name.split(".")[0] for file_name in audio_file_list]
        self.analysis_folder = analysis_folder
        self.transform = transform
        self.time_window = time_window  # now using 60 seconds as default

        # key labels
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        self.key_mapping = {keys[i]: i for i in range(24)}

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, idx):
        audio_file_name = self.audio_file_list[idx]
        audio_file_path = os.path.join(self.audio_folder, audio_file_name)
        audio_features = self.pair_audio_with_features(audio_file_name)

        audio_data, sr = librosa.load(audio_file_path, sr=22050)
        frame_length = int(self.time_window * sr)
        num_frames = len(audio_data) // frame_length
        audio_data = audio_data[:num_frames * frame_length]  # only keep the first num_frames * frame_length samples
        chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)
        key, bpm = audio_features[audio_file_name]
        key_label = self.key_to_label(key)

        # also returing bpm?
        return chromagram, key_label

    # pair each file with its annotations
    def pair_audio_with_features(self, file_name):
        analysis_full_path = os.path.join(self.analysis_folder, file_name) + "_analysis.json"
        key = ""
        bpm = -1
        with open(analysis_full_path) as f:
            annotations = json.load(f)
            key = annotations["tonality"]
            bpm = annotations["tempo"]
            f.close()

        return {file_name: [key, bpm]}

    def key_to_label(self, key):
        return self.key_mapping.get(key, -1)

