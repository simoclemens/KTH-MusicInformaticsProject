from key_extraction.data_loader import AudioDataLoader, AudioDataset
import json
import os
import numpy as np
import random

audio_path = "FSL10K/audio/wav"
analysis_path = "FSL10K/ac_analysis"

def complete_audio_file_name(uncomplete_file_name):
    audios_file_list = os.listdir(audio_path)
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

        print(len(techno_audios))

        with open(techno_audios_file_path, "w") as f:
            json.dump(techno_audios, f)
            f.close()
        f.close()
    return techno_audios


def split_train_test_data(data, split_ratio=0.8):
    num_samples = len(data)
    num_train_samples = int(num_samples * split_ratio)
    random.shuffle(data)

    train_data = data[:num_train_samples]
    test_data = data[num_train_samples:]

    return train_data, test_data


def write_train_test_indexes(train_data, test_data, train_index_file="train_indices.json", test_index_file="test_indices.json"):
    train_indexes = [dataset.audio_file_list.index(filename) for filename in train_data]
    test_indexes = [dataset.audio_file_list.index(filename) for filename in test_data]

    train_features = [dataset.pair_audio_with_features(filename)[filename] for filename in train_data]
    test_features = [dataset.pair_audio_with_features(filename)[filename] for filename in test_data]
    
    # save np array in json file?
    train_data_dict = {"indices": train_indexes, "labels": train_features}
    test_data_dict = {"indices": test_indexes, "labels": test_features}

    with open(train_index_file, "w") as f:
        json.dump(train_data_dict, f)

    with open(test_index_file, "w") as f:
        json.dump(test_data_dict, f)


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
    dataset = AudioDataset(audio_path, analysis_path, time_window=60)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)
    techno_audios = extract_techno_audios("parent_genres.json", "techno_audios.json")
    # write_techno_features_to_file(techno_audios, dataset)
    train_data, test_data = split_train_test_data(techno_audios)
    write_train_test_indexes(train_data, test_data)
