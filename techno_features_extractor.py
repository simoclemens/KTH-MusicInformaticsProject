from data_loader import AudioDataLoader, AudioDataset
import json
import os
import random

audio_path = "FSL10K/audio/wav"
analysis_path = "FSL10K/ac_analysis"
min_target_duration = 5
max_target_duration = 30


def complete_audio_file_name(uncomplete_file_name):
    audios_file_list = os.listdir(audio_path)
    audios_file_list = [file_name.split(".")[0] for file_name in audios_file_list]
    for file_name in audios_file_list:
        if uncomplete_file_name in file_name:
            return file_name
    
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


def split_train_test_data(data, split_ratio=0.8):
    num_samples = len(data)
    num_train_samples = int(num_samples * split_ratio)
    random.shuffle(data)

    train_data = data[:num_train_samples]
    test_data = data[num_train_samples:]

    return train_data, test_data


def get_audio_duration_in_seconds(filename):
    file_path = os.path.join(audio_path, filename + '.wav.wav')
    duration = -1
    if os.path.exists(file_path):
        duration = AudioDataset.get_audio_duration(file_path)
    return duration


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


def write_train_test_indices(train_data, test_data, train_index_file="train_indices.json", test_index_file="test_indices.json"):
    train_indexes = [dataset.audio_file_list.index(filename) for filename in train_data]
    test_indexes = [dataset.audio_file_list.index(filename) for filename in test_data]

    labels = [{"duration_seconds": get_audio_duration_in_seconds(filename)
               for filename, index in zip(dataset.audio_file_list, range(len(dataset.audio_file_list)))},
              {"key": dataset.pair_audio_with_features(filename)[filename][0] for filename, index in zip(dataset.audio_file_list, range(len(dataset.audio_file_list)))},
              {"bpm": dataset.pair_audio_with_features(filename)[filename][1] for filename, index in zip(dataset.audio_file_list, range(len(dataset.audio_file_list)))}]
    
    train_data_dict = {"indexes": train_indexes, "labels": labels}
    test_data_dict = {"indexes": test_indexes, "labels": labels}
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
    techno_audios = write_techno_audios("parent_genres.json", "techno_audios.json")
    selected_techno_audios = write_selected_techno_audios_with_duration(techno_audios, "selected_techno_audios.json")
    print(len(selected_techno_audios))

    # write_techno_features_to_file(selected_techno_audios)
    # train_data, test_data = split_train_test_data(selected_techno_audios)
    # write_train_test_indices(train_data, test_data)

