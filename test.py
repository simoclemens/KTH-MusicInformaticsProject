import librosa
from beatfinder import nnBeatExtraction

path = "samples\\test1.wav"
bpm, beat = nnBeatExtraction(path)
print(bpm)
print(beat)
