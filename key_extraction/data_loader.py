import os
import json
from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
import librosa
from shutil import copyfile 

audio_path = "FSL10K/audio/wav"
output_path = "output_folder"  # Change this to the desired output folder
techno_audios_list = "key_extraction/techno_audios.json"


class AudioDataset(Dataset):
    def __init__(self, audio_folder, analysis_folder, transform=None, time_window=30):
        self.audio_folder = audio_folder
        audio_file_list = os.listdir(audio_folder)
        self.audio_file_list = [file_name.split(".")[0] for file_name in audio_file_list if file_name.endswith(".wav.wav")] 
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
        audio_data, sr = librosa.load(audio_file_name, duration = self.time_window, sr=22050)
        frame_length = int(self.time_window * sr)
        num_frames = len(audio_data) // frame_length
        audio_data = audio_data[:num_frames * frame_length]  # only keep the first num_frames * frame_length samples
        chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=2048, hop_length=512)

        audio_features = self.pair_audio_with_features(audio_file_name)
        key, bpm = audio_features[audio_file_name]
        key_label = self.key_to_label(key)

        # also returing bpm?
        return chromagram, key_label

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

    def key_to_label(self, key):
        return self.key_mapping.get(key, -1)

    def get_audio_duration(audio_path):
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0 
        return duration_seconds

    def save_techno_audios(self, output_folder, target_duration):
        os.makedirs(output_folder, exist_ok=True)

        # Step 1: Read the list of selected audio filenames from techno_audios.json
        with open(techno_audios_list, "r") as f:
            selected_audio_filenames = json.load(f)

        for audio_filename in selected_audio_filenames:
            audio_file_path = os.path.join(self.audio_folder, audio_filename + ".wav.wav")

            if os.path.exists(audio_file_path):
                duration_seconds = AudioDataset.get_audio_duration(audio_file_path)

                if duration_seconds > target_duration:
                    audio = AudioSegment.from_wav(audio_file_path)
                    trimmed_audio = audio[:target_duration * 1000]
                    output_file_path = os.path.join(output_folder, audio_filename + ".wav")
                    trimmed_audio.export(output_file_path, format="wav")

                elif duration_seconds <= target_duration:
                    output_file_path = os.path.join(output_folder, audio_filename + ".wav")
                    copyfile(audio_file_path, output_file_path)



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