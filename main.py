from tracks import TrackFeatures
from nn_procedure import NNMixing
import librosa

track_list = []

samples1, sr1 = librosa.load("test1.wav")
samples2, sr2 = librosa.load("test2.wav")
samples3, sr3 = librosa.load("test3.wav")

track1 = TrackFeatures("test1.wav", samples1, sr1)
track2 = TrackFeatures("test2.wav", samples2, sr2)
track3 = TrackFeatures("test3.wav", samples3, sr3)

track1.extractBeat()
track1.extractKey()
track2.extractBeat()
track2.extractKey()
track3.extractBeat()
track3.extractKey()

track_list.append(track1)
track_list.append(track2)
track_list.append(track3)

print(track1.bpm, track2.bpm, track3.bpm)
print(track1.key, track2.key, track3.key)
print(track1.beat)

mixer = NNMixing(track_list)
