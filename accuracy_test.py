import os
import json
from tracks import TrackFeatures
import sklearn as sk
from data_loader import AudioDataLoader, AudioDataset

audio_path = "FSL10K/audio/wav/"
analysis_path = "FSL10K/ac_analysis"
data_json = "generated_json/test_indices.json"
data_output_json = "generated_json/accuracy_data.json"
final_output_json = "generated_json/accuracy_result.json"

def accuracy_tests(filename):
    file_path = os.path.join(audio_path, filename)
    track = TrackFeatures(file_path, key_mode= "determ",beat_mode="dynamic")
    track.extractKey()
    track.extractBeat()
    determ_key = track.key
    dynamic_bpm = track.bpm
    track = TrackFeatures(file_path, key_mode="nn", beat_mode="nn")
    track.extractKey()
    track.extractBeat()
    nn_key = track.key
    nn_bpm = track.bpm

    # key accuracy
    actual_key = get_key_and_bpm(data_json, filename)[0]

    accuracy_determ = sk.metrics.accuracy_score([actual_key], [determ_key])
    accuracy_nn = sk.metrics.accuracy_score([actual_key], [nn_key])

    # bpm mse
    actual_bpm = get_key_and_bpm(data_json, filename)[1]

    mse_dynamic = sk.metrics.mean_squared_error([actual_bpm], [dynamic_bpm])
    mse_nn = sk.metrics.mean_squared_error([actual_bpm], [nn_bpm])

    return actual_key, determ_key, nn_key, accuracy_determ, accuracy_nn, \
        actual_bpm, dynamic_bpm, nn_bpm, mse_dynamic, mse_nn


def compute_accuracy():
    dataset = AudioDataset(audio_path, analysis_path)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)

    key_accuracies_determ = []
    key_accuracies_nn = []
    bpm_mses_dynamic = []
    bpm_mses_nn = []

    selected_files = json.load(open(data_json, "r"))
    print("Number of selected files: ", len(selected_files))
    # for file in selected_files:

    json_output = []
    for file in selected_files:
        filename = file["filename"]
        complete_filename = audio_path+filename
        act_key, key_det, key_nn, key_acc_det, key_acc_nn,\
            act_bpm, bpm_dyn, bpm_nn, bpm_mse_dynamic, bpm_mse_nn = accuracy_tests(filename)

        key_accuracies_determ.append(key_acc_det)
        key_accuracies_nn.append(key_acc_nn)
        bpm_mses_dynamic.append(bpm_mse_dynamic)
        bpm_mses_nn.append(bpm_mse_nn)
        json_output.append({"filename": filename, "duration": AudioDataset.get_audio_duration(complete_filename), 
                            "actual_key": act_key, "key_determ": key_det, "key_nn": key_nn, 
                            "actual_bpm": act_bpm, "bpm_dynamic": bpm_dyn, "bpm_nn": bpm_nn})
        
        
    with open(data_output_json, "w") as f:
        json.dump(json_output, f)
        f.close()

    key_accuraciy_determ = sum(key_accuracies_determ) / len(key_accuracies_determ)
    key_accuraciy_nn = sum(key_accuracies_nn) / len(key_accuracies_nn)
    bpm_mse_dynamic = sum(bpm_mses_dynamic) / len(bpm_mses_dynamic)
    bpm_mse_nn = sum(bpm_mses_nn) / len(bpm_mses_nn)

    return key_accuraciy_determ, key_accuraciy_nn, bpm_mse_dynamic, bpm_mse_nn


def get_key_and_bpm(data_json, filename):
    files = json.load(open(data_json, "r"))
    for file in files:
        if file["filename"] == filename:
            return file["key"], file["bpm"]


if __name__ == "__main__":
    key_accuracy_determ, key_accuracy_nn, bpm_mse_dynamic, bpm_mse_nn = compute_accuracy()
    print("Key accuracy determ: ", key_accuracy_determ)
    print("Key accuracy nn: ", key_accuracy_nn)
    print("BPM mse dynamic: ", bpm_mse_dynamic)
    print("BPM mse nn: ", bpm_mse_nn)

    with open(final_output_json, "w") as f:
        json.dump({"key_accuracy_determ": key_accuracy_determ, "key_accuracy_nn": key_accuracy_nn, "bpm_mse_dynamic": bpm_mse_dynamic, "bpm_mse_nn": bpm_mse_nn}, f)
        f.close()
    

