import librosa
import librosa.display
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor, detect_beats
from madmom.features.tempo import interval_histogram_acf, detect_tempo


def dynamicBeatExtraction(samples, sr):
    bpm, beat = librosa.beat.beat_track(y=samples, sr=sr)
    beat = librosa.frames_to_time(beat, sr=sr)

    return bpm, beat


def nnBeatExtraction(file_path,duration):
    beat_activations = RNNBeatProcessor()(file_path)
    # beat = detect_beats(beat_activations, 10)

    beat_tracker = BeatTrackingProcessor(fps=100)
    beat = beat_tracker(beat_activations)

    # interval_histogram = interval_histogram_acf(beat_activations)
    # bpm = detect_tempo(interval_histogram, fps=100)[0][0]
    duration_min = duration/60
    bpm = len(beat) / duration_min


    return bpm, beat


