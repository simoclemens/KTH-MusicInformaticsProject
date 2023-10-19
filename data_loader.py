import os
import json
from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
import librosa
from shutil import copyfile 

output_path = "output_folder"  # Change this to the desired output folder
techno_audios_list = "generated_json/selected_techno_audios.json"

class AudioDataset(Dataset):
    def __init__(self, audio_folder, analysis_folder, transform=None, time_window=30):
        self.audio_folder = audio_folder
        audio_file_list = os.listdir(audio_folder)
        self.audio_file_list = [file_name.split(".")[0] for file_name in audio_file_list if file_name.endswith(".wav.wav")] 
        self.analysis_folder = analysis_folder
        self.transform = transform
        self.time_window = time_window  # now using 60 seconds as default

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, idx):
        audio_file_name = self.audio_folder + self.audio_file_list[idx] + ".wav.wav"
        samples, sr = self.getsamples(audio_file_name)
        audio_features = self.pair_audio_with_features(audio_file_name)
        key, bpm = audio_features[audio_file_name]
        key_label = self.key_to_label(key)

        # also returing bpm?
        return audio_file_name, sr, samples, key_label, bpm

    def load_audio(self, filename):
        samples, sr = librosa.load(filename)
        print("File: ", filename, "Sample rate: ", sr, "Samples: ", len(samples))
        return samples, sr

    def getsamples(self, filename):
        samples, sr = librosa.load(filename)
        print("File: ", filename, "Sample rate: ", sr, "Samples: ", len(samples))
        frame_length = int(self.time_window * sr)
        num_frames = len(samples) // frame_length
        samples = samples[:num_frames * frame_length]

        return samples, sr
        

    # pair each file with its annotations
    def pair_audio_with_features(self, file_name):
        analysis_full_path = os.path.join(self.analysis_folder, file_name) + "_analysis.json"
        key = ""
        bpm = -1
        if os.path.exists(analysis_full_path):
            with open(analysis_full_path) as f:
                annotations = json.load(f)
                key = annotations["tonality"]
                bpm = annotations["tempo"]
                f.close()

        return {file_name: [key, bpm]}


    def key_to_label(key):
        key_mapping_sharp, key_mapping_flat = AudioDataset.get_key_mapping()
        
        if key in key_mapping_sharp:
            return key_mapping_sharp[key]
        elif key in key_mapping_flat:
            return key_mapping_flat[key]
        else:
            print("Key not found: ", key)
            return -1

    def get_key_mapping():
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        key_mapping_sharp = {keys[i]: i for i in range(24)}

        pitches = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        key_mapping_flat = {keys[i]: i for i in range(24)}
        return key_mapping_sharp, key_mapping_flat

    def get_audio_duration(audio_path):
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0 
        return duration_seconds

    def save_techno_audios(self, output_folder, max_target_duration=30):
        copied_files = 0
        os.makedirs(output_folder, exist_ok=True)

        # Step 1: Read the list of selected audio filenames from techno_audios.json
        with open(techno_audios_list, "r") as f:
            selected_audio_filenames = json.load(f)
        print(len(selected_audio_filenames))

        for audio_filename in selected_audio_filenames:
            audio_file_path = os.path.join(self.audio_folder, audio_filename + ".wav.wav")

            if os.path.exists(audio_file_path):
                duration_seconds = AudioDataset.get_audio_duration(audio_file_path)

                if duration_seconds > max_target_duration:
                    audio = AudioSegment.from_wav(audio_file_path)
                    trimmed_audio = audio[:max_target_duration * 1000]
                    output_file_path = os.path.join(output_folder, audio_filename + ".wav")
                    trimmed_audio.export(output_file_path, format="wav")
                    copied_files += 1

                else:
                    output_file_path = os.path.join(output_folder, audio_filename + ".wav")
                    copyfile(audio_file_path, output_file_path)
                    copied_files += 1
        print("Copied {} files".format(copied_files))


class AudioDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=collate_fn)


if __name__ == "__main__":
    # test
    duration = 30
    dataset = AudioDataset("FSL10K\\audio\\wav", "ac_analysis", time_window=duration)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)
    output_folder = "selected_techno_audios"
    dataset.save_techno_audios(output_folder, duration)

