from key_extraction.data_loader import AudioDataLoader, AudioDataset
import json
import os

def complete_audio_file_name(uncomplete_file_name, audios_folder="audio"):
    audios_file_list = os.listdir(audios_folder)
    audios_file_list = [file_name.split(".")[0] for file_name in audios_file_list]
    for file_name in audios_file_list:
        if uncomplete_file_name in file_name:
            return file_name
    
def extract_techno_audios(genres_file_path, techno_audios_file_path="techno_audios.json"):
    with open(genres_file_path) as f:
        audios_with_genres = json.load(f)
        techno_audios = []
        for audio, genres in audios_with_genres.items():
            complete_file_name = complete_audio_file_name(audio)
            for genre in genres:
                if "techno" in genre and not complete_file_name is None: # (if full dataset is not downloaded, some audios may not be present)
                    techno_audios.append(complete_file_name)

        with open(techno_audios_file_path, "w") as f:
            json.dump(techno_audios, f)
            f.close()
        f.close()
    return techno_audios

def write_techno_features_to_file(techno_audios, dataset, techno_features_file_name = "techno_audios_features.json"):
    audio_files_list = dataset.audio_file_list
    techno_audios_features= []
    for audio in audio_files_list:
        with open("techno_audios.json") as f:
            techno_audios = json.load(f)
            
            # match audio with its features and write in file
            if audio in techno_audios:
                audio_features = dataset.pair_audio_with_features(audio)
                techno_audios_features.append(audio_features)
    with open(techno_features_file_name, "w") as f:
        json.dump(techno_audios_features, f)
        f.close()

if __name__ == "__main__":
    dataset = AudioDataset("audio", "ac_analysis", time_window=60)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)
    techno_audios = extract_techno_audios("parent_genres.json", "techno_audios.json")
    write_techno_features_to_file(techno_audios, dataset)