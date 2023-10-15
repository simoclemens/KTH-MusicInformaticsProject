from tracks import TrackFeatures
import numpy as np
import EQ_utilities, os, librosa
from pydub import AudioSegment

def overwriteAndGetTrackAndAudioSegment(audio, filename):
   audio.export(filename, format='wav')
   sample, sr = librosa.load(filename)
   trackFeatures = TrackFeatures(filename, sample, sr)
   audioSegment = AudioSegment.from_wav(trackFeatures.file_name)
   return trackFeatures , audioSegment

class Mixer:
  
    def __init__(self, playlistFolder):
        self.playlistFolder = playlistFolder

        self.tracksFeatures: [TrackFeatures] = []
        self.modifiedTracksFolder = self.playlistFolder + '/modifiedTracksPlaylist/'
        self.__createTracksFeaturesList(self.tracksFeatures, self.playlistFolder+'/')     

    def __createTracksFeaturesList(self, tracksFeatures, destination):
       for trackFile in os.listdir(destination):
         file = destination + trackFile
         if not os.path.isdir(file):
           sample, sr = librosa.load(file)
           trackFeat = TrackFeatures(file, sample, sr)
           trackFeat.extractFeatures()
           tracksFeatures.append(trackFeat)
       tracksFeatures = sorted(tracksFeatures, key=lambda x: x.bpm) 

    def createMix(self, secondsOfTransition):
       self.modifiedTracksFeatures: [TrackFeatures] = []
       #self.__createTracksFeaturesList(self.modifiedTracksFeatures, self.playlistFolder)
       track1 = self.tracksFeatures[0]
       track2 = self.tracksFeatures[1]
       self.__createFolder(self.modifiedTracksFolder)
       self.mixTwoTracks(track1=track1, track2=track2, secondsOfTransition=secondsOfTransition)


    def mixTwoTracks(self, track1: TrackFeatures, track2: TrackFeatures, secondsOfTransition):
       audio1 = EQ_utilities.change_tempo(track1, 125)
       audio2 = EQ_utilities.change_tempo(track2, 128)
       #overwrite original file
       destinationAudio1 = self.modifiedTracksFolder + track1.file_name.split('/')[1]
       destinationAudio2 = self.modifiedTracksFolder + track2.file_name.split('/')[1]
       
       track1, audio1 = overwriteAndGetTrackAndAudioSegment(audio1, destinationAudio1)
       track2, audio2 = overwriteAndGetTrackAndAudioSegment(audio2, destinationAudio2)
       track1.extractFeatures()
       track2.extractFeatures()
       track2_initial_instant_for_transition = track2.beat[0]*1000
       audio2 = audio2[track2_initial_instant_for_transition:]
       
       out_track_exiting_instant, index = self.closest(track1.beat, secondsOfTransition, True)
        
       audio1, info = EQ_utilities.gradual_tempo_change(track1, final_tempo=128, final_segment=out_track_exiting_instant)
       print(track1.bpm, track2.bpm)
      #re-analize speeded up track
      #  audio1.export(self.modifiedTracksFolder+'bpmIncreased.wav', format="wav")
      #  sample, sr = librosa.load(self.modifiedTracksFolder+'bpmIncreased.wav')
      #  audio1SpeededUp = TrackFeatures(self.modifiedTracksFolder+'bpmIncreased.wav', sample, sr)
      #  audio1SpeededUp.extractFeatures()
      #  print("index:", index)
      #  out_track_exiting_instant = audio1SpeededUp.beat[index]*1000

       audio1_exiting_section = audio1[out_track_exiting_instant:]
       audio1_without_exiting_section = audio1[:out_track_exiting_instant]
       audio1_exiting_section = EQ_utilities.time_varying_high_pass_filter(audio1_exiting_section, initial_cutoff=1, final_cutoff=400, duration=secondsOfTransition)
       audio1_exiting_section = audio1_exiting_section.fade_out(secondsOfTransition*1000)
       audio1 = audio1_without_exiting_section + audio1_exiting_section
       
       audio2_entering_section = audio2[:secondsOfTransition*1000]
       audio2_without_entering_section = audio2[secondsOfTransition*1000:]
       audio2_entering_section = EQ_utilities.time_varying_high_pass_filter(audio2_entering_section, initial_cutoff=400, final_cutoff=1, duration=secondsOfTransition)
       audio2_entering_section = audio2_entering_section.fade_in(secondsOfTransition*1000)
       audio2 = audio2_entering_section + audio2_without_entering_section
       
       audio1.export(self.modifiedTracksFolder+'audio1final_'+'mixed.wav', format="wav")
       audio2.export(self.modifiedTracksFolder+'audio2final_'+'mixed.wav', format="wav")
       
       silent_section_for_overlay = AudioSegment.silent(duration=audio2.duration_seconds*1000)
       audio1 += silent_section_for_overlay
       mixedAudio = audio1.overlay(audio2, position=out_track_exiting_instant)
       mixedAudio.export(self.modifiedTracksFolder+'final_'+'mixed.wav', format="wav")

    def closest(self, lst, secondsTransition, fromBack):
      if fromBack:
        instant = lst[-1]-secondsTransition
      else:
        instant = lst[0]+secondsTransition
      lst = np.asarray(lst)
      idx = (np.abs(lst - instant)).argmin()
      return lst[idx]*1000, idx

    def __createFolder(self, folderName):
        if not os.path.isdir(folderName):
          os.makedirs(folderName)
        else:
          for filename in os.listdir(folderName):
            print(filename)
            os.remove(f"{folderName}/{filename}")

          
mixer = Mixer('playlist')
mixer.createMix(20) #s










