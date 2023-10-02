from keyfinder import TonalFragment
class TrackFeatures:
    def __init__(self, name, samples, sr):
        self.name = name
        self.samples = samples
        self.sr = sr
        self.selected = False
        self.key = None
        self.bpm = None

    def setSelected(self):
        self.selected = True
