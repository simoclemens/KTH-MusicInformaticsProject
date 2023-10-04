import librosa
audio1, sr1 = librosa.load("Q1.wav")
bpm, beat = librosa.beat.beat_track(y=audio1, sr=sr1)
print(bpm)
print(beat)
