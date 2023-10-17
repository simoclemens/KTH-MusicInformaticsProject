from data_loader import AudioDataLoader, AudioDataset
import json
import os
import random
from musicnn.extractor import extractor

audio_path = "FSL10K/audio/wav"
analysis_path = "FSL10K/ac_analysis"
min_target_duration = 6
max_target_duration = 30


def complete_audio_file_name(uncomplete_file_name):
    audios_file_list = os.listdir(audio_path)
    audios_file_list = [file_name.split(".")[0] for file_name in audios_file_list]
    for file_name in audios_file_list:
        if uncomplete_file_name in file_name:
            return file_name

def extract_musicnn_features_with_duration(audio_file, duration_in_seconds):
    taggram, tags, features = extractor(audio_path+"/"+audio_file, model='MTT_musicnn',input_length=duration_in_seconds, extract_features=True)
    features_pen = features['penultimate']
    features_pen = features_pen.tolist()
    return features_pen

def extract_musicnn_features(audio_file):
    taggram, tags, features = extractor(audio_path+"/"+audio_file, model='MTT_musicnn', extract_features=True)
    features_pen = features['penultimate']
    features_pen_list = features_pen.tolist()
    print("Features shape: ", len(features_pen))
    
    return features_pen_list


def get_audio_duration_in_seconds(filename):
    file_path = os.path.join(audio_path, filename + '.wav.wav')
    duration = -1
    if os.path.exists(file_path):
        duration = AudioDataset.get_audio_duration(file_path)
    return duration

def split_train_test_data(data, split_ratio=0.8):
    num_samples = len(data)
    num_train_samples = int(num_samples * split_ratio)
    random.shuffle(data)

    train_data = data[:num_train_samples]
    test_data = data[num_train_samples:]

    return train_data, test_data

def write_techno_audios(genres_file_path, techno_audios_file_path="techno_audios.json"):
    with open(genres_file_path) as f:
        audios_with_genres = json.load(f)
        techno_audios = []
        for audio, genres in audios_with_genres.items():
            complete_file_name = complete_audio_file_name(audio)
            for genre in genres:
                # (if full dataset is not downloaded, some audios may not be present; if file has wrong format, complete_file_name will be None)
                if "techno" in genre and not complete_file_name is None and not complete_file_name in techno_audios: 
                    techno_audios.append(complete_file_name)

        print(len(techno_audios))

        with open(techno_audios_file_path, "w") as f:
            json.dump(techno_audios, f)
            f.close()
        f.close()
    return techno_audios

def write_selected_techno_audios_with_duration(techno_audios, sel_techno_audios_file = "selected_techno_audios.json", techno_audios_with_duration_file="techno_audios_with_duration.json"):
    selected_techno_audios_with_duration = []
    for audio in techno_audios:
        audio_duration = get_audio_duration_in_seconds(audio)
        if audio_duration > min_target_duration:
            if audio_duration > max_target_duration:
                audio_duration = max_target_duration
            selected_techno_audios_with_duration.append({audio: audio_duration})
    with open(techno_audios_with_duration_file, "w") as f:
        json.dump(selected_techno_audios_with_duration, f)
        f.close()
    with open(sel_techno_audios_file, "w") as f:
        json.dump([list(audio.keys())[0] for audio in selected_techno_audios_with_duration], f)
        f.close()
    return [list(audio.keys())[0] for audio in selected_techno_audios_with_duration]


def write_train_test_indices(train_data, test_data, train_index_file="train_indices_features.json", test_index_file="test_indices_features.json"):
    print("Train data: ", len(train_data), "Test data: ", len(test_data))

    train_data_dict = [{"index": dataset.audio_file_list.index(file),
                    "filename": file + ".wav.wav",
                    "duration_seconds": get_audio_duration_in_seconds(file),
                    "key": dataset.pair_audio_with_features(file)[file][0], 
                    "bpm": dataset.pair_audio_with_features(file)[file][1],
                    "features": extract_musicnn_features(str(file + ".wav.wav")),
                    } for file in train_data
                    ]
    
    test_data_dict = [{
                    "index": dataset.audio_file_list.index(file),
                    "filename": file + ".wav.wav",
                    "duration_seconds": get_audio_duration_in_seconds(file),
                    "key": dataset.pair_audio_with_features(file)[file][0], 
                    "bpm": dataset.pair_audio_with_features(file)[file][1],
                    "features": extract_musicnn_features(str(file + ".wav.wav")),
                    } for file in test_data
                    ]
    
    with open(train_index_file, "w") as f:
        json.dump(train_data_dict, f)

    with open(test_index_file, "w") as f:
        json.dump(test_data_dict, f)

def write_techno_features_to_file(techno_audios, techno_features_file_name = "selected_techno_audios_features.json"):
    techno_audios_features= []
    for audio in techno_audios:
        audio_features = dataset.pair_audio_with_features(audio)
        techno_audios_features.append(audio_features)
    with open(techno_features_file_name, "w") as f:
        json.dump(techno_audios_features, f)
        f.close()

if __name__ == "__main__":
    time_window = 30
    dataset = AudioDataset(audio_path, analysis_path, time_window=time_window)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)
    
    techno_audios = json.load(open("generated_json/techno_audios.json", "r"))
    selected_techno_audios = write_selected_techno_audios_with_duration(techno_audios)
    print(len(selected_techno_audios))

    write_techno_features_to_file(selected_techno_audios)
    train_data, test_data = split_train_test_data(selected_techno_audios)
    write_train_test_indices(train_data, test_data)


        