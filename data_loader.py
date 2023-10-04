import os
import json
from torch.utils.data import Dataset, DataLoader
import librosa

# Define a custom dataset class
class AudioLoader(DataLoader):
    def __init__(self, audio_folder, analysis_folder, transform=None):
        self.audio_folder = audio_folder
        self.analysis_folder = analysis_folder
        self.transform = transform

        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        self.key_mapping = {keys[i]: i for i in range(24)}

    def pair_audio_with_features(self, file_name):
        analysis_full_path = os.path.join(self.analysis_folder, file_name) + "_analysis.json"
        key = ""
        bpm = -1
        with open(analysis_full_path) as f:
            annotations = json.load(f)
            key = annotations["tonality"]
            bpm = annotations["tempo"]
            f.close()

        # Return dictionary instance with filename as key and analysis as value
        return {file_name: [key, bpm]}

    def key_to_label(self,key):
        return self.key_mapping.get(key, -1)  # Return -1 if the key is not found in the mapping

if __name__ == "__main__":
    loader = AudioLoader("audio/wav", "ac_analysis")
