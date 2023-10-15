import librosa
import librosa.display
#from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
#from madmom.features.tempo import interval_histogram_acf, detect_tempo


def dynamicBeatExtraction(samples, sr):
    bpm, beat = librosa.beat.beat_track(y=samples, sr=sr)

    return bpm, beat


# def nnBeatExtraction(file_path):
#     beat_activations = RNNBeatProcessor()(file_path)
#     beat_tracker = BeatTrackingProcessor(fps=100)
#     beat = beat_tracker(beat_activations)

#     interval_histogram = interval_histogram_acf(beat_activations)
#     bpm = detect_tempo(interval_histogram, fps=100)[0][0]

#     return bpm, beat


