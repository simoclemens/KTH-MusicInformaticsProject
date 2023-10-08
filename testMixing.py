from tracks import TrackFeatures
import librosa

samples1, sr1 = librosa.load("test1.wav")
track1 = TrackFeatures("NeverGonnaGiveYouUp.wav", samples1, sr1)

print(track1.extractFeatures())