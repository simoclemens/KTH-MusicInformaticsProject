import librosa
from beatfinder import nnBeatExtraction, dynamicBeatExtraction

path = "samples\\test1.wav"
samples, sr = librosa.load(path)
bpm, beat = dynamicBeatExtraction(samples,sr)
print(bpm)
print(beat)
