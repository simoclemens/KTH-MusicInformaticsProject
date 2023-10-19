import os
import json
from tracks import TrackFeatures
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
from data_loader import AudioDataLoader, AudioDataset

audio_path = "FSL10K/audio/wav/"
analysis_path = "FSL10K/ac_analysis"
data_json = "generated_json/train_test_features_15s/test_indices_features.json"
data_output_json = "generated_json/accuracy_data_15s.json"
final_output_json = "generated_json/accuracy_result_15s.json"

def accuracy_tests(filename):
    file_path = os.path.join(audio_path, filename)
    track = TrackFeatures(file_path, key_mode= "determ", beat_mode="dynamic")
    track.extractFeatures()
    determ_key = track.key
    dynamic_bpm = track.bpm
    track = TrackFeatures(file_path, key_mode="nn", beat_mode="nn")
    track.extractFeatures()
    nn_key = track.key
    nn_bpm = track.bpm
    track = TrackFeatures(file_path, key_mode="tl")
    track.extractFeatures()
    tl_key = track.key

    # key accuracy
    actual_key = get_key_and_bpm(data_json, filename)[0]

    accuracy_determ = sk.metrics.accuracy_score([actual_key], [determ_key])
    accuracy_nn = sk.metrics.accuracy_score([actual_key], [nn_key])
    accuracy_tl = sk.metrics.accuracy_score([actual_key], [tl_key])

    # bpm mse
    actual_bpm = get_key_and_bpm(data_json, filename)[1]

    mse_dynamic = sk.metrics.mean_squared_error([actual_bpm], [dynamic_bpm])
    mse_nn = sk.metrics.mean_squared_error([actual_bpm], [nn_bpm])

    return actual_key, determ_key, nn_key, tl_key, accuracy_determ, accuracy_nn, accuracy_tl, \
        actual_bpm, dynamic_bpm, nn_bpm, mse_dynamic, mse_nn


def compute_accuracy():
    dataset = AudioDataset(audio_path, analysis_path)
    dataloader = AudioDataLoader(dataset, batch_size=32, shuffle=False)

    key_accuracies_determ = []
    key_accuracies_nn = []
    key_accuracies_tl = []
    bpm_mses_dynamic = []
    bpm_mses_nn = []

    selected_files = json.load(open(data_json, "r"))
    print("Number of selected files: ", len(selected_files))
    # for file in selected_files:

    json_output = []
    for file in selected_files:
        filename = file["filename"]
        complete_filename = audio_path+filename
        act_key, key_det, key_nn, key_tl, key_acc_det, key_acc_nn, key_acc_tl,\
            act_bpm, bpm_dyn, bpm_nn, bpm_mse_dynamic, bpm_mse_nn = accuracy_tests(filename)

        key_accuracies_determ.append(key_acc_det)
        key_accuracies_nn.append(key_acc_nn)
        key_acc_tl.append(key_acc_tl)
        bpm_mses_dynamic.append(bpm_mse_dynamic)
        bpm_mses_nn.append(bpm_mse_nn)
        json_output.append({"filename": filename, "duration": AudioDataset.get_audio_duration(complete_filename), 
                            "actual_key": act_key, "key_determ": key_det, "key_determ_acc" : key_acc_det, "key_nn": key_nn, "key_nn_acc": key_acc_nn, 
                            "actual_bpm": act_bpm, "bpm_dynamic": bpm_dyn, "bpm_mse_dynamic": bpm_mse_dynamic, "bpm_nn": bpm_nn, "bpm_mse_nn": bpm_mse_nn})        
        
    with open(data_output_json, "w") as f:
        json.dump(json_output, f)
        f.close()

    key_accuracy_determ = sum(key_accuracies_determ) / len(key_accuracies_determ)
    key_accuracy_nn = sum(key_accuracies_nn) / len(key_accuracies_nn)
    key_accuracy_tl = sum(key_accuracies_tl) / len(key_accuracies_tl)
    bpm_mse_dynamic = sum(bpm_mses_dynamic) / len(bpm_mses_dynamic)
    bpm_mse_nn = sum(bpm_mses_nn) / len(bpm_mses_nn)

    return key_accuracy_determ, key_accuracy_nn, key_accuracy_tl, bpm_mse_dynamic, bpm_mse_nn


def get_key_and_bpm(data_json, filename):
    files = json.load(open(data_json, "r"))
    for file in files:
        if file["filename"] == filename:
            key = file["key"]
            bpm = file["bpm"]
            return key, bpm

def plot_accuracy():
    accuracy_data = json.load(open(data_output_json, "r"))
    bin_size = 1
    duration_bins = np.arange(15, 31, bin_size)

    mean_key_determ_acc = []
    mean_key_nn_acc = []
    mean_key_tl_acc = []
    mean_bpm_mse_dynamic = []
    mean_bpm_mse_nn = []
    marker_sizes = []  # List to store marker sizes

    for start_duration in duration_bins:
        end_duration = start_duration + bin_size
        accuracies_in_bin = [item for item in accuracy_data if start_duration <= item["duration"] < end_duration]
        num_data_points = len(accuracies_in_bin)
        marker_sizes.append(20 + num_data_points * 2)  # Adjust the marker size based on data points

        if accuracies_in_bin:
            mean_key_determ_acc.append(np.mean([item["key_determ_acc"] for item in accuracies_in_bin]))
            mean_key_nn_acc.append(np.mean([item["key_nn_acc"] for item in accuracies_in_bin]))
            mean_key_tl_acc.append(np.mean([item["key_tl_acc"] for item in accuracies_in_bin]))
            mean_bpm_mse_dynamic.append(np.mean([item["bpm_mse_dynamic"] for item in accuracies_in_bin]))
            mean_bpm_mse_nn.append(np.mean([item["bpm_mse_nn"] for item in accuracies_in_bin]))
        else:
            mean_key_determ_acc.append(0)
            mean_key_nn_acc.append(0)
            mean_key_tl_acc.append(0)
            mean_bpm_mse_dynamic.append(0)
            mean_bpm_mse_nn.append(0)

    # Create scatter plots for accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(duration_bins, mean_key_determ_acc, label='Key Deterministic Accuracy', s=marker_sizes, marker='o', alpha=0.7)
    plt.scatter(duration_bins, mean_key_nn_acc, label='Key NN Accuracy', s=marker_sizes, marker='o', alpha=0.7)
    plt.scatter(duration_bins, mean_key_tl_acc, label='Key TL Accuracy', s=marker_sizes, marker='o', alpha=0.7)
    plt.plot(duration_bins, mean_key_determ_acc, linestyle='-', linewidth=2, markersize=0, color='blue', alpha=0.7)
    plt.plot(duration_bins, mean_key_nn_acc, linestyle='-', linewidth=2, markersize=0, color='orange', alpha=0.7)
    plt.plot(duration_bins, mean_key_tl_acc, linestyle='-', linewidth=2, markersize=0, color='green', alpha=0.7)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy vs. Duration Bins')
    plt.legend()
    plt.grid()
    plt.show()

    # Create scatter plots for MSE
    plt.figure(figsize=(10, 6))
    plt.scatter(duration_bins, mean_bpm_mse_dynamic, label='BPM MSE (Dynamic)', s=marker_sizes, marker='o', alpha=0.7)
    plt.scatter(duration_bins, mean_bpm_mse_nn, label='BPM MSE (NN)', s=marker_sizes, marker='o', alpha=0.7)
    plt.plot(duration_bins, mean_bpm_mse_dynamic, linestyle='-', linewidth=2, markersize=0, color='blue', alpha=0.7)
    plt.plot(duration_bins, mean_bpm_mse_nn, linestyle='-', linewidth=2, markersize=0, color='orange', alpha=0.7)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Mean MSE')
    plt.title('Mean MSE vs. Duration Bins')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    key_accuracy_determ, key_accuracy_nn, key_accuracy_tl, bpm_mse_dynamic, bpm_mse_nn = compute_accuracy()
    print("Key accuracy determ: ", key_accuracy_determ)
    print("Key accuracy nn: ", key_accuracy_nn)
    print("Key accuracy tl: ", key_accuracy_tl)
    print("BPM mse dynamic: ", bpm_mse_dynamic)
    print("BPM mse nn: ", bpm_mse_nn)

    with open(final_output_json, "w") as f:
        json.dump({"key_accuracy_determ": key_accuracy_determ, "key_accuracy_nn": key_accuracy_nn, "key_accuracy_tl": key_accuracy_tl,
                   "bpm_mse_dynamic": bpm_mse_dynamic, "bpm_mse_nn": bpm_mse_nn}, f)
        f.close()

    plot_accuracy()

    

