from functools import reduce
from tracks import TrackFeatures
import numpy as np
import EQ_utilities, os, librosa
from pydub import AudioSegment

def overwriteAndGetTrackAndAudioSegment(audio, filename):
   audio.export(filename, format='wav')
   sample, sr = librosa.load(filename)
   trackFeatures = TrackFeatures(filename, sample, sr)
   trackFeatures.extractFeatures()
   audioSegment = AudioSegment.from_wav(trackFeatures.file_name)
   return trackFeatures, audioSegment

class Mixer:

    def __init__(self, trackPlaylist, playlistFolder='playlist/'):
        self.trackPlaylist = trackPlaylist
        self.playlistFolder = playlistFolder
        self.modifiedTracksFolder = self.playlistFolder + 'modifiedTracksPlaylist/'
        self.mixedSegments = []  # Store all the mixed and non-mixed segments
        prev_transition_end = 0
        for idx, track in enumerate(self.trackPlaylist[:-1]):
            track1 = track
            track2 = self.trackPlaylist[idx + 1]
            
            audio1Exiting, audio2Entering, in_transition_start_ms, out_transition_end_ms, track2_initial_ms_for_transition = self.createMix(track1=track1, track2=track2)
            audio1Exiting.export(self.modifiedTracksFolder + str(idx)+'audio1Exiting.wav', format="wav")
            audio2Entering.export(self.modifiedTracksFolder +  str(idx)+'audio2Entering.wav', format="wav")
            self.mixedSegments.append(audio1Exiting[prev_transition_end:in_transition_start_ms])

            prev_transition_end = out_transition_end_ms

            l1 = len(audio1Exiting[in_transition_start_ms:])
            l2 = len(audio2Entering[:out_transition_end_ms])
            print(l1, l2, track2_initial_ms_for_transition)
            silentSegmentToExtendAudio1 = AudioSegment.silent(duration=abs(l1-l2)-track2_initial_ms_for_transition)
            audio1Exiting += silentSegmentToExtendAudio1
            # Transition
            transition_mix = audio1Exiting[in_transition_start_ms:].overlay(audio2Entering[:out_transition_end_ms], 0)
            self.mixedSegments.append(transition_mix)

        lastAudio = AudioSegment.from_file(track2.file_name)
        last_track_remainder = lastAudio[out_transition_end_ms:]
        self.mixedSegments.append(last_track_remainder)
        
        full_mix = reduce(lambda segment1, segment2: segment1 + segment2, self.mixedSegments)
        full_mix.export(self.modifiedTracksFolder + 'full_mix.wav', format="wav")

    def createMix(self, track1, track2, secondsOfTransition=10):
        self.__createFolder(self.modifiedTracksFolder)
        return self.mixTwoTracks(track1=track1, track2=track2, secondsOfTransition=secondsOfTransition)

    def mixTwoTracks(self, track1: TrackFeatures, track2: TrackFeatures, secondsOfTransition):

      track1.beat = librosa.frames_to_time(track1.beat, sr=track1.sr) #now beats are at the second
      track2.beat = librosa.frames_to_time(track2.beat, sr=track2.sr)

      audio1 = AudioSegment.from_file(track1.file_name)
      audio2 = AudioSegment.from_file(track2.file_name)
      track2_initial_ms_for_transition = track2.beat[0]
      audio2 = audio2[track2_initial_ms_for_transition:]

      # Get the approximate transition instant
      out_track_exiting_second, _ = self.closest(track1.beat, secondsOfTransition, True)
      print(out_track_exiting_second)
      # Apply gradual tempo change to track1
      audio1 = EQ_utilities.gradual_tempo_change(track1, final_tempo=track2.bpm, final_second_for_tempo_increase=out_track_exiting_second)
      # Analyze the part of track1 after the final_segment for exact transition instant
      out_track_exit_instant_in_ms = out_track_exiting_second*1000
      adjusted_audio1_segment = audio1[out_track_exit_instant_in_ms:]
      adjusted_track1_filename = "temp_adjusted_track1.wav"
      track1_adjusted, _ = overwriteAndGetTrackAndAudioSegment(adjusted_audio1_segment, adjusted_track1_filename)
      track1_adjusted.beat = librosa.frames_to_time(track1_adjusted.beat, sr=track1_adjusted.sr) #now beats are at the second
      relative_out_track_exiting_second, _ = self.closest(track1_adjusted.beat, secondsOfTransition, True)
      # Need to add to the previous one to account for the relative out track instant
      exact_out_track_exiting_second = out_track_exiting_second + relative_out_track_exiting_second
      print('Exact: ',exact_out_track_exiting_second)
      # Handle audio1 segments

      audio1_exiting_section = audio1[exact_out_track_exiting_second*1000:]
      audio1_without_exiting_section = audio1[:exact_out_track_exiting_second*1000]
      audio1_exiting_section = EQ_utilities.time_varying_high_pass_filter(audio1_exiting_section, initial_cutoff=1, final_cutoff=400, duration=secondsOfTransition)
      audio1_exiting_section = audio1_exiting_section.fade_out(secondsOfTransition*1000)
      audio1 = audio1_without_exiting_section + audio1_exiting_section

      # Handle audio2 segments
      audio2_entering_section = audio2[:secondsOfTransition*1000]
      audio2_without_entering_section = audio2[secondsOfTransition*1000:]
      audio2_entering_section = EQ_utilities.time_varying_high_pass_filter(audio2_entering_section, initial_cutoff=400, final_cutoff=1, duration=secondsOfTransition)
      audio2_entering_section = audio2_entering_section.fade_in(secondsOfTransition*1000)
      audio2 = audio2_entering_section + audio2_without_entering_section

      # Clean up temporary files
      if os.path.exists(adjusted_track1_filename):
          os.remove(adjusted_track1_filename)

      transition_end_instant_relative_to_track2_ms = secondsOfTransition * 1000

      return audio1, audio2, exact_out_track_exiting_second*1000, transition_end_instant_relative_to_track2_ms, track2_initial_ms_for_transition

    def closest(self, lst, secondsTransition, fromBack):
        #print(lst, secondsTransition)
        if fromBack:
            instant = lst[-1] - secondsTransition
        else:
            instant = lst[0] + secondsTransition
        lst = np.asarray(lst)
        idx = (np.abs(lst - instant)).argmin()
        return lst[idx], idx

    def __createFolder(self, folderName):
        if not os.path.isdir(folderName):
            os.makedirs(folderName)
        else:
            for filename in os.listdir(folderName):
                os.remove(os.path.join(folderName, filename))


def loadTrack(filename):
   sample, sr = librosa.load(filename)
   trackFeatures = TrackFeatures(filename, sample, sr)
   trackFeatures.extractFeatures()
   return trackFeatures


from tracks import TrackFeatures
# TO DO
track_list = []

for track in track_list:
    track.extractFeatures()

track_selection = [
                    loadTrack("playlist/Estasi (mp3cut.net)(1).wav"),
                    loadTrack("playlist/AeroMex (mp3cut.net).wav"),
                    loadTrack("playlist/Inside Out (mp3cut.net).wav")
                ]
#track_lineup = track_selection.#()

mixer = Mixer(track_selection)
#mixer.createMix()

