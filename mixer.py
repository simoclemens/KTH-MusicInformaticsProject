from tracks import TrackFeatures
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import os
import librosa
from pydub import AudioSegment
import pyrubberband
from pydub.effects import speedup
from pydub.playback import play

keys = [
        ['E major', 'B major', 'F# major', 'D♭ major', 'A♭ major', 'E♭ major', 'B♭ major', 'F major', 'C major', 'G major', 'D major', 'A major'],
        ['D♭ minor', 'A♭ minor', 'E♭ minor', 'B♭ minor', 'F minor', 'C minor', 'G minor', 'D minor', 'A minor', 'E minor', 'B minor', 'F# minor']
    ]

def percentageDifference(bpm1, bpm2):
    return abs(bpm1- bpm2)/((bpm1+bpm2)/2.0)*100.0

def calculateManhattanDistance(key1, key2):
    key_indices1 = [(i, j) for i, row in enumerate(keys) for j, k in enumerate(row) if k == key1]
    key_indices2 = [(i, j) for i, row in enumerate(keys) for j, k in enumerate(row) if k == key2]
    row_diff = abs(key_indices1[0][0]-key_indices2[0][0])
    col_diff = min(abs(key_indices1[0][1]-key_indices2[0][1]), len(keys[0])-abs(key_indices1[0][1]-key_indices2[0][1]))
    return row_diff+col_diff 

def findNeighbouringKeys(key): #based on the Chamelot Wheel
    neighboring_keys = []

    key_indices = [(i, j) for i, row in enumerate(keys) for j, k in enumerate(row) if k == key]

    directionsRow1 = [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1)]
    directionsRow2 = [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1)]

    if key_indices:
        i, j = key_indices[0] 

        if i == 0:  # Key is in the first row
            for dr, dc in directionsRow1:
                new_y_pos, new_x_pos = i+dr, j+dc
                neighboring_keys.append(keys[new_y_pos][new_x_pos])
        else:  # Key is in the second row
            for dr, dc in directionsRow2:
                new_y_pos, new_x_pos = i+dr, j+dc
                neighboring_keys.append(keys[new_y_pos][new_x_pos])

    return neighboring_keys

class Mixer:
    def __init__(self, playlistFolder):
        self.playlistFolder = playlistFolder
        self.tracksFeatures: [TrackFeatures] = []
        self.modifiedTracksFolder = self.playlistFolder + '/modifiedTracksPlaylist/'
        self.__createTracksFeaturesList(self.tracksFeatures, self.playlistFolder+'/')
        for trackFile in os.listdir(playlistFolder):
          file = playlistFolder + '/' + trackFile
          if not os.path.isdir(file):
            sample, sr = librosa.load(file)
            trackFeat = TrackFeatures(file, sample, sr)
            trackFeat.extractFeatures()
            self.tracksFeatures.append(trackFeat)
        
        self.tracksFeatures = sorted(self.tracksFeatures, key=lambda x: x.bpm)
        self.alignTracksBpmWithFastest()      

    def __createTracksFeaturesList(self, tracksFeatures, destination):
       for trackFile in os.listdir(destination):
         file = destination + trackFile
         if not os.path.isdir(file):
           sample, sr = librosa.load(file)
           trackFeat = TrackFeatures(file, sample, sr)
           trackFeat.extractFeatures()
           tracksFeatures.append(trackFeat)
       tracksFeatures = sorted(tracksFeatures, key=lambda x: x.bpm) 

    def alignTracksBpmWithFastest(self):
       fastestBpm = self.tracksFeatures[-1].bpm
       self.__createFolder(self.modifiedTracksFolder)
       speed_incrase_percentage_per_track = [percentageDifference(track.bpm, fastestBpm) for track in self.tracksFeatures]
       for track_index, track in enumerate(self.tracksFeatures):
          speedchange = speed_incrase_percentage_per_track[track_index]
          filename = track.file_name.split('/')[1]
          if speedchange != 0:
            timeChangeCoeff = 1+speed_incrase_percentage_per_track[track_index]/100.0
          else:
             timeChangeCoeff = 1
          audio = AudioSegment.from_wav(track.file_name)
          if timeChangeCoeff != 1:
            new_file = speedup(audio,timeChangeCoeff,timeChangeCoeff*100)
          else:
            new_file = audio
          new_file.export(self.modifiedTracksFolder+'modified_'+filename, format="wav")

    def createMix(self, secondsOfTransition):
       self.modifiedTracksFeatures: [TrackFeatures] = []
       self.__createTracksFeaturesList(self.modifiedTracksFeatures, self.modifiedTracksFolder)
       out_track_exiting_instant = self.closest(self.modifiedTracksFeatures[0].beat, secondsOfTransition, True)
       print(out_track_exiting_instant)
       crossfade = (self.modifiedTracksFeatures[0].duration - out_track_exiting_instant)*1000
       audio1 = AudioSegment.from_wav(self.modifiedTracksFeatures[0].file_name)[:self.modifiedTracksFeatures[0].beat[-1]*1000]
       audio2 = AudioSegment.from_wav(self.modifiedTracksFeatures[1].file_name)[self.modifiedTracksFeatures[1].beat[0]*1000:]
       audio3_mixed = audio1.append(audio2, crossfade)
       audio3_mixed.export(self.modifiedTracksFolder+'modified_'+'mixed.wav', format="wav")
       #print([[(beat) for beat in track.beat] for track in self.modifiedTracksFeatures])

    def closest(self, lst, secondsTransition, fromBack):
     if fromBack:
      instant = lst[-1]-secondsTransition
     else:
      instant = lst[0]+secondsTransition
     lst = np.asarray(lst)
     idx = (np.abs(lst - instant)).argmin()
     return lst[idx]

    def __createFolder(self, folderName):
        if not os.path.isdir(folderName):
          os.makedirs(folderName)
        else:
          for filename in os.listdir(folderName):
            print(filename)
            os.remove(f"{folderName}/{filename}")
            
          
mixer = Mixer('playlist')
mixer.createMix(5)










