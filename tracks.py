from keyfinder import determKeyExtraction, nnKeyExtraction
from beatfinder import dynamicBeatExtraction, nnBeatExtraction


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
        self.key_label = None
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
            bpm, beat = dynamicBeatExtraction(self.samples, self.sr)
        elif self.beat_mode == "nn":
            bpm, beat = nnBeatExtraction(self.file_name)

        self.bpm = bpm
        self.beat = beat

    def extractKey(self):
        if self.key_mode == "determ":
            key = determKeyExtraction(self.samples, self.sr)
        elif self.key_mode == "nn":
            key = nnKeyExtraction(self.file_name)

        self.key = key



