from keyfinder import determKeyExtraction, nnKeyExtraction, tlKeyExtraction
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
        self.key_idx = None
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
        elif self.key_mode == "tl":
            self.key = tlKeyExtraction(self.file_name)

        key_dict = {'C minor': 0, 'C major': 1,
                    'C# minor': 2, 'Db minor': 2, 'C# major': 3, 'Db major': 3,
                    'D minor': 4, 'D major': 5,
                    'D# minor': 6, 'Eb minor': 6, 'D# major': 7, 'Eb major': 7,
                    'E minor': 8, 'E major': 9,
                    'F minor': 10, 'F major': 11,
                    'F# minor': 12, 'Gb minor': 12, 'F# major': 13, 'Gb major': 13,
                    'G minor': 14, 'G major': 15,
                    'G# minor': 16, 'Ab minor': 16, 'G# major': 17, 'Ab major': 17,
                    'A minor': 18, 'A major': 19,
                    'A# minor': 20, 'Bb minor': 20, 'A# major': 21, 'Bb major': 21,
                    'B minor': 22, 'B major': 23}

        self.key_idx = key_dict[self.key]




