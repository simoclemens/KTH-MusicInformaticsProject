import os
import json
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
from data_loader import AudioDataLoader, AudioDataset
from tracks import TrackFeatures

# Define file paths
audio_path = "FSL10K/audio/wav/"
analysis_path = "FSL10K/ac_analysis"
data_json = "generated_json/train_test_features_25s/test_indices_features_with_duration.json"
data_output_json = "generated_json/accuracy_data_25s.json"
final_output_json = "generated_json/accuracy_result_25s.json"

# Function to perform accuracy tests for a given audio file
def accuracy_tests(filename):
    # Load the audio file
    file_path = os.path.join(audio_path, filename)
    # Extract features using different key and beat modes
    track = TrackFeatures(file_path, key_mode="determ", beat_mode="dynamic")
    track.extractFeatures()
    determ_key = track.key
    dynamic_bpm = track.bpm
    track = TrackFeatures(file_path, key_mode="nn", beat_mode="nn")
    track.extractFeatures()
    nn_key = track.key
    nn_bpm = track.bpm

    # Key accuracy
    actual_key, actual_bpm = get_key_and_bpm(data_json, filename)
    accuracy_determ = sk.metrics.accuracy_score([actual_key], [determ_key])
    accuracy_nn = sk.metrics.accuracy_score([actual_key], [nn_key])
    
    # Calculate top-K key accuracy
    topk_key = top_k_key_accuracy(actual_key, [determ_key, nn_key], k=3)

    # Confusion matrices
    confusion_matrix_determ = sk.metrics.confusion_matrix([actual_key], [determ_key])
    confusion_matrix_nn = sk.metrics.confusion_matrix([actual_key], [nn_key])

    key_acc_dict = {
        "actual_key": actual_key,
        "determ_key": determ_key,
        "nn_key": nn_key,
        "accuracy_determ": accuracy_determ,
        "accuracy_nn": accuracy_nn,
        "topk_key": topk_key,
        "confusion_matrix_determ": confusion_matrix_determ.tolist(),
        "confusion_matrix_nn": confusion_matrix_nn.tolist()
    }

    # BPM Mean Squared Error
    mse_dynamic = sk.metrics.mean_squared_error([actual_bpm], [dynamic_bpm])
    mse_nn = sk.metrics.mean_squared_error([actual_bpm], [nn_bpm])

    bpm_mse_dict = {
        "actual_bpm": actual_bpm,
        "dynamic_bpm": dynamic_bpm,
        "nn_bpm": nn_bpm,
        "mse_dynamic": mse_dynamic,
        "mse_nn": mse_nn
    }

    return key_acc_dict, bpm_mse_dict

# Function to calculate confusion matrix metrics
def calculate_confusion_matrix_metrics(key_acc_dict, threshold=0.5):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # Extract values from the confusion matrices
    confusion_matrix_determ = key_acc_dict["confusion_matrix_determ"]
    confusion_matrix_nn = key_acc_dict["confusion_matrix_nn"]

    true_positives_determ = confusion_matrix_determ[1][1]
    true_negatives_determ = confusion_matrix_determ[0][0]
    false_positives_determ = confusion_matrix_determ[0][1]
    false_negatives_determ = confusion_matrix_determ[1][0]

    true_positives_nn = confusion_matrix_nn[1][1]
    true_negatives_nn = confusion_matrix_nn[0][0]
    false_positives_nn = confusion_matrix_nn[0][1]
    false_negatives_nn = confusion_matrix_nn[1][0]

    # Accumulate values
    total_true_positives += true_positives_determ + true_positives_nn
    total_true_negatives += true_negatives_determ + true_negatives_nn
    total_false_positives += false_positives_determ + false_positives_nn
    total_false_negatives += false_negatives_determ + false_negatives_nn

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = (total_true_positives + total_true_negatives) / (
        total_true_positives + total_true_negatives + total_false_positives + total_false_negatives
    )

    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)

    # F1-score is the harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

# Function to calculate dataset-level metrics
def calculate_dataset_metrics(json_output):
    actual_keys = [item["key"] for item in json_output]
    determ_keys = [item["key_determ"] for item in json_output]
    nn_keys = [item["key_nn"] for item in json_output]

    confusion_matrix_actual_vs_determ = sk.metrics.confusion_matrix(actual_keys, determ_keys)
    confusion_matrix_actual_vs_nn = sk.metrics.confusion_matrix(actual_keys, nn_keys)

    accuracy = (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_determ[0][0]
        + confusion_matrix_actual_vs_nn[1][1]
        + confusion_matrix_actual_vs_nn[0][0]
    ) / (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_determ[0][0]
        + confusion_matrix_actual_vs_determ[0][1]
        + confusion_matrix_actual_vs_determ[1][0]
        + confusion_matrix_actual_vs_nn[1][1]
        + confusion_matrix_actual_vs_nn[0][0]
        + confusion_matrix_actual_vs_nn[0][1]
        + confusion_matrix_actual_vs_nn[1][0]
    )

    precision = (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_nn[1][1]
    ) / (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_determ[0][1]
        + confusion_matrix_actual_vs_nn[1][1]
        + confusion_matrix_actual_vs_nn[0][1]
    )

    recall = (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_nn[1][1]
    ) / (
        confusion_matrix_actual_vs_determ[1][1]
        + confusion_matrix_actual_vs_determ[1][0]
        + confusion_matrix_actual_vs_nn[1][1]
        + confusion_matrix_actual_vs_nn[1][0]
    )

    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

# Function to calculate key similarity
def key_similarity(key1, key2):
    key_mapping = {
        # Major keys
        "C major": 0, "C# major": 1, "D major": 2, "D# major": 3, "E major": 4,
        "F major": 5, "F# major": 6, "G major": 7, "G# major": 8, "A major": 9,
        "A# major": 10, "B major": 11,

        # Minor keys
        "C minor": 1.5, "C# minor": 2.5, "D minor": 3.5, "D# minor": 4.5, "E minor": 5.5,
        "F minor": 6.5, "F# minor": 7.5, "G minor": 8.5, "G# minor": 9.5, "A minor": 10.5,
        "A# minor": 11.5, "B minor": 12.5
    }

    key1_value = key_mapping.get(key1, -1)
    key2_value = key_mapping.get(key2, -1)

    if key1_value == -1 or key2_value == -1:
        return 0.0
    similarity = 1.0 / (1.0 + abs(key1_value - key2_value))

    return similarity

# Function to calculate top-K key accuracy
def top_k_key_accuracy(actual_key, detected_keys, k=3):
    similarity_scores = [key_similarity(actual_key, detected_key) for detected_key in detected_keys]
    sorted_detected_keys = [x for _, x in sorted(zip(similarity_scores, detected_keys), reverse=True)]
    top_k_detected_keys = sorted_detected_keys[:k]
    return actual_key in top_k_detected_keys

# Function to compute accuracy metrics for the entire dataset
def compute_accuracy():
    dataset = AudioDataset(audio_path, analysis_path)

    key_accuracies_determ = []
    key_accuracies_nn = []
    topk_accuracies = []
    bpm_mses_dynamic = []
    bpm_mses_nn = []

    selected_files = json.load(open(data_json, "r"))
    print("Number of selected files: ", len(selected_files))

    json_output = []
    for file in selected_files:
        filename = file["filename"]
        complete_filename = os.path.join(audio_path, filename)
        key_acc_dict, bpm_mse_dict = accuracy_tests(filename)

        key_accuracies_determ.append(key_acc_dict["accuracy_determ"])
        key_accuracies_nn.append(key_acc_dict["accuracy_nn"])
        topk_accuracies.append(key_acc_dict["topk_key"])
        bpm_mses_dynamic.append(bpm_mse_dict["mse_dynamic"])
        bpm_mses_nn.append(bpm_mse_dict["mse_nn"])
        conf_matrix_metrics = calculate_confusion_matrix_metrics(key_acc_dict)

        json_dict = {
            "filename": filename,
            "key": key_acc_dict["actual_key"],
            "bpm": bpm_mse_dict["actual_bpm"],
            "key_determ": key_acc_dict["determ_key"],
            "key_nn": key_acc_dict["nn_key"],
            "topk_key": key_acc_dict["topk_key"],
            "key_confusion_matrix_determ": key_acc_dict["confusion_matrix_determ"],
            "key_confusion_matrix_nn": key_acc_dict["confusion_matrix_nn"],
            "key_confusion_matrix_metrics": conf_matrix_metrics,
            "key_accuracy_determ": key_acc_dict["accuracy_determ"],
            "key_accuracy_nn": key_acc_dict["accuracy_nn"],
            "bpm_dynamic": bpm_mse_dict["dynamic_bpm"],
            "bpm_nn": bpm_mse_dict["nn_bpm"],
            "bpm_mse_dynamic": bpm_mse_dict["mse_dynamic"],
            "bpm_mse_nn": bpm_mse_dict["mse_nn"]
        }
        json_output.append(json_dict)

    with open(data_output_json, "w") as f:
        json.dump(json_output, f)
        f.close()

    accuracy, precision, recall, f1_score = calculate_dataset_metrics(json_output)
    key_accuracy_determ = sum(key_accuracies_determ) / len(key_accuracies_determ)
    key_accuracy_nn = sum(key_accuracies_nn) / len(key_accuracies_nn)
    mean_topk_accuracy = np.mean(topk_accuracies)
    bpm_mse_dynamic = sum(bpm_mses_dynamic) / len(bpm_mses_dynamic)
    bpm_mse_nn = sum(bpm_mses_nn) / len(bpm_mses_nn)

    return key_accuracy_determ, key_accuracy_nn, mean_topk_accuracy, bpm_mse_dynamic, bpm_mse_nn, accuracy, precision, recall, f1_score

# Function to get key and BPM from data JSON
def get_key_and_bpm(data_json, filename):
    files = json.load(open(data_json, "r"))
    for file in files:
        if file["filename"] == filename:
            key = file["key"]
            bpm = file["bpm"]
            return key, bpm

# Function to plot accuracy
def plot_accuracy():
    accuracy_6s_2080 = json.load(open("generated_json/accuracy/accuracy_result_6s.json", "r"))
    accuracy_10s_2080 = json.load(open("generated_json/accuracy/accuracy_result_10s.json", "r"))
    accuracy_15s_filter = json.load(open("generated_json/accuracy/accuracy_result_15s.json", "r"))
    accuracy_25s_filter = json.load(open("generated_json/accuracy/accuracy_result_25s.json", "r"))

    key_accuracies_determ = [
        accuracy_6s_2080["key_accuracy_determ"],
        accuracy_10s_2080["key_accuracy_determ"],
        accuracy_15s_filter["key_accuracy_determ"],
        accuracy_25s_filter["key_accuracy_determ"]
    ]
    key_accuracies_nn = [
        accuracy_6s_2080["key_accuracy_nn"],
        accuracy_10s_2080["key_accuracy_nn"],
        accuracy_15s_filter["key_accuracy_nn"],
        accuracy_25s_filter["key_accuracy_nn"]
    ]

    bpm_mses_dynamic = [
        accuracy_6s_2080["bpm_mse_dynamic"],
        accuracy_10s_2080["bpm_mse_dynamic"],
        accuracy_15s_filter["bpm_mse_dynamic"],
        accuracy_25s_filter["bpm_mse_dynamic"]
    ]
    bpm_mses_nn = [
        accuracy_6s_2080["bpm_mse_nn"],
        accuracy_10s_2080["bpm_mse_nn"],
        accuracy_15s_filter["bpm_mse_nn"],
        accuracy_25s_filter["bpm_mse_nn"]
    ]

    x = ["80:20 (>6s)", "80:20 (>10s)", ">6s : >15s", ">6s : >25s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(x, key_accuracies_determ, label="Key accuracy (deterministic)", marker='o', color='c')
    ax1.plot(x, key_accuracies_nn, label="Key accuracy (neural network)", marker='o', color='m')
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Key Accuracy vs. Dataset Split")
    ax1.set_xticks(x)
    ax1.legend()

    ax2.plot(x, bpm_mses_dynamic, label="BPM mse (dynamic)", marker='o', color='g')
    ax2.plot(x, bpm_mses_nn, label="BPM mse (neural network)", marker='o', color='m')
    ax2.set_xlabel("Train:Test")
    ax2.set_ylabel("MSE")
    ax2.set_title("BPM Mean Squared Error vs. Dataset Split")
    ax2.set_xticks(x)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("generated_json/accuracy/accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    # key_accuracy_determ, key_accuracy_nn, mean_topk_accuracy, bpm_mse_dynamic, bpm_mse_nn, accuracy, precision, recall, f1_score = compute_accuracy()

    # # Write on json
    # with open(final_output_json, "w") as f:
    #     json.dump({"key_accuracy_determ": key_accuracy_determ, "key_accuracy_nn": key_accuracy_nn, "mean_topk_accuracy": mean_topk_accuracy,
    #                 "bpm_mse_dynamic": bpm_mse_dynamic, "bpm_mse_nn": bpm_mse_nn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}, f)
    #     f.close()

    plot_accuracy()

    

