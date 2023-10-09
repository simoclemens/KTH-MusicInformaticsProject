import librosa
from beatfinder import nnBeatExtraction, dynamicBeatExtraction
from keyfinder import nnKeyExtraction, determKeyExtraction

path = "samples\\test1.wav"
samples, sr = librosa.load(path)
bpm, beat = dynamicBeatExtraction(samples, sr)
key = determKeyExtraction(samples, sr)
print(bpm)
print(beat)
print(key)
bpm, beat = nnBeatExtraction(path)
key = nnKeyExtraction(path)
print(bpm)
print(beat)
print(key)
