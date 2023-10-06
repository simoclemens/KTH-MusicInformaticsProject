from keyfinder import TonalFragment
import librosa
from BeatNet.BeatNet import BeatNet
import numpy as np


class TrackFeatures:
    def __init__(self, file_name, samples, sr, key_mode="determ", beat_mode="dynamic"):
        self.file_name = file_name
        self.samples = samples
        self.sr = sr
        self.key_mode = key_mode
        self.beat_mode = beat_mode
        self.duration = len(samples) / sr
        self.selected = False
        self.key = None
        self.second_key = None
        self.bpm = None
        self.beat = None

    def setSelected(self):
        self.selected = True

    def extractFeatures(self):
        self.extractBeat()
        self.extractKey()

    def extractBeat(self):
        bpm = None
        beat = None

        if self.beat_mode == "dynamic":
            bpm, beat = self.dynamicBeatExtraction()
        elif self.beat_mode == "nn":
            pass

        self.bpm = bpm
        self.beat = beat

    def extractKey(self):
        if self.key_mode == "determ":
            self.determExtractKey()
        elif self.key_mode == "nn":
            pass

    def dynamicBeatExtraction(self):
        bpm, beat = librosa.beat.beat_track(y=self.samples, sr=self.sr)

        return bpm, beat

    def nnBeatExtraction(self, model=1):
        estimator = BeatNet(model, mode='offline', inference_model='DBN', plot=[], thread=False, device="Cuda")
        output = estimator.process("audio file directory")

        duration_min = self.duration / 60
        self.bpm = output.shape[0] / duration_min
        self.beat = output[:, 0]

    def determExtractKey(self):
        # compute chromograph
        chromograph = librosa.feature.chroma_cqt(y=self.samples, sr=self.sr, bins_per_octave=24)

        # chroma_vals is the amount of each pitch class present in this time interval
        chroma_vals = []
        for i in range(12):
            chroma_vals.append(np.sum(chromograph[i]))
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # dictionary relating pitch names to the associated intensity in the song
        keyfreqs = {pitches[i]: chroma_vals[i] for i in range(12)}

        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        min_key_corrs = []
        maj_key_corrs = []
        for i in range(12):
            key_test = [keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
            min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

        # names of all major and minor keys
        key_dict = {**{keys[i]: maj_key_corrs[i] for i in range(12)},
                    **{keys[i + 12]: min_key_corrs[i] for i in range(12)}}

        # this attribute represents the key determined by the algorithm
        self.key = max(key_dict, key=key_dict.get)
        best_corr = max(key_dict.values())

        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.second_key = None
        second_best_corr = None

        for key, corr in key_dict.items():
            if corr > best_corr * 0.9 and corr != best_corr:
                if second_best_corr is None or corr > second_best_corr:
                    self.second_key = key
                    second_best_corr = corr




