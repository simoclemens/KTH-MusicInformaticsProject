from keyfinder import determKeyExtraction, nnKeyExtraction
from beatfinder import dynamicBeatExtraction, nnBeatExtraction
import librosa


class TrackFeatures:
    def __init__(self, file_name, key_mode="determ", beat_mode="dynamic"):
        self.file_name = file_name
        self.samples, self.sr = librosa.load(file_name)
        self.key_mode = key_mode
        self.beat_mode = beat_mode
        self.duration = len(self.samples) / self.sr
        self.selected = False
        self.key = None
        self.key_label = None
        self.second_key = None
        self.bpm = None
        self.beat = None
        self.out_exiting_instant = None
        self.mod_track = self.samples

    def setSelected(self):
        self.selected = True

    def extractFeatures(self):
        self.extractBeat()
        self.extractKey()
        print("TRACK: "+self.file_name+" -> "+"BPM:"+str(self.bpm)+" KEY:"+self.key)


    def extractBeat(self):
        bpm = None
        beat = None

        if self.beat_mode == "dynamic":
            bpm, beat = dynamicBeatExtraction(self.samples, self.sr)
        elif self.beat_mode == "nn":
            bpm, beat = nnBeatExtraction(self.file_name, self.duration)

        self.bpm = bpm
        self.beat = beat

    def extractKey(self):
        if self.key_mode == "determ":
            self.key = determKeyExtraction(self.samples, self.sr)
        elif self.key_mode == "nn":
            self.key = nnKeyExtraction(self.file_name)



