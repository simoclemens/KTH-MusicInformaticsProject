from functools import reduce
from tracks import TrackFeatures
import numpy as np
import EQ_utilities, os, librosa
from pydub import AudioSegment


def overwriteAndGetTrackAndAudioSegment(audio, filename, beat_mode):
    audio.export(filename, format='wav')
    trackFeatures = TrackFeatures(filename, beat_mode=beat_mode)
    trackFeatures.extractFeatures()
    audioSegment = AudioSegment.from_wav(trackFeatures.file_name)
    return trackFeatures, audioSegment


class Mixer:

    def __init__(self, trackPlaylist, beat_mode='dynamic', playlistFolder='playlist/'):
        self.trackPlaylist = trackPlaylist
        self.playlistFolder = playlistFolder
        self.beat_mode = beat_mode
        self.modifiedTracksFolder = self.playlistFolder + 'modifiedTracksPlaylist/'
        self.mixedSegments = []  # Store all the mixed and non-mixed segments

    def mixPlaylist(self):
        prev_transition_end_ms = 0
        for idx, track in enumerate(self.trackPlaylist[:-1]):
            print('IDX:', idx)
            track1 = track
            track2 = self.trackPlaylist[idx + 1]
            audio1Exiting, audio2Entering, in_transition_start_ms, out_transition_end_ms, \
                track2_initial_ms_for_transition = self.createMix(track1=track1, track2=track2, secondsOfTransition=10)

            # Remember: previous 'track2' is now the current one ([idx]), and previous one was shifted by track2_initial_ms_for_transition[idx-1]
            nonMixedSection = audio1Exiting[prev_transition_end_ms:in_transition_start_ms]
            self.mixedSegments.append(nonMixedSection)

            # The silent section is a mismatch caused by the track2_initial_ms_for_transition I think, even though it's really small
            # Update
            prev_transition_end_ms = out_transition_end_ms
            # this avoids silent sections to form
            diff = len(audio1Exiting[in_transition_start_ms:]) - len(audio2Entering[:out_transition_end_ms])
            # Transition
            if diff > 0:
                transition_mix = (audio1Exiting[in_transition_start_ms:-diff]).overlay(
                    audio2Entering[:out_transition_end_ms], position=0)
            else:
                transition_mix = (audio1Exiting[in_transition_start_ms:]).overlay(
                    audio2Entering[:out_transition_end_ms - diff], position=0)

            self.mixedSegments.append(transition_mix)

        lastAudio = AudioSegment.from_file(track2.file_name)
        last_track_remainder = lastAudio[out_transition_end_ms:]
        self.mixedSegments.append(last_track_remainder)

        full_mix = reduce(lambda segment1, segment2: segment1 + segment2, self.mixedSegments)
        full_mix.export(self.modifiedTracksFolder + 'full_mix.wav', format="wav")

    def createMix(self, track1, track2, secondsOfTransition=18):
        self.__createFolder(self.modifiedTracksFolder)
        return self.mixTwoTracks(track1=track1, track2=track2, secondsOfTransition=secondsOfTransition)

    def mixTwoTracks(self, track1: TrackFeatures, track2: TrackFeatures, secondsOfTransition):

        audio2 = AudioSegment.from_file(track2.file_name)
        track2_initial_ms_for_transition = track2.beat[0] * 1000
        audio2 = audio2[track2_initial_ms_for_transition:]

        # Get the approximate transition instant
        out_track_exiting_second, _ = self.closest(track1.beat, secondsOfTransition, True)

        # Apply gradual tempo change to track1
        audio1, final_segment_length_ms = EQ_utilities.gradual_tempo_change(track1, final_tempo=track2.bpm,
                                                                            final_second_for_tempo_increase=out_track_exiting_second)

        # Analyze the part of track1 after the final_segment for exact transition instant
        adjusted_audio1_segment = audio1[len(audio1) - final_segment_length_ms:]
        adjusted_track1_filename = "temp_adjusted_track1.wav"
        track1_adjusted, _ = overwriteAndGetTrackAndAudioSegment(adjusted_audio1_segment, adjusted_track1_filename,
                                                                 self.beat_mode)
        relative_out_track_exiting_second, _ = self.closest(track1_adjusted.beat, 0, False)

        # Need to add to the previous one to account for the relative out track instant
        out_track_exiting_section_start_seconds = (len(audio1[:-final_segment_length_ms])) / 1000
        exact_out_track_exiting_second = out_track_exiting_section_start_seconds + relative_out_track_exiting_second

        # Handle audio1 segments
        audio1_without_exiting_section = audio1[:exact_out_track_exiting_second * 1000]
        audio1_exiting_section = audio1[exact_out_track_exiting_second * 1000:]

        # fade out low frequencies in a gradual way
        audio1_exiting_section = EQ_utilities.time_varying_high_pass_filter(audio1_exiting_section, initial_cutoff=1,
                                                                            final_cutoff=400,
                                                                            duration=len(audio1_exiting_section) / 1000)
        # apply the volume fade out to also other frequencies
        audio1_exiting_section = audio1_exiting_section.fade_out(len(audio1_exiting_section))
        audio1 = audio1_without_exiting_section + audio1_exiting_section

        # Handle audio2 segments
        audio2_entering_section = audio2[:secondsOfTransition * 1000]
        audio2_without_entering_section = audio2[secondsOfTransition * 1000:]

        # fade in low frequencies in a gradual way
        audio2_entering_section = EQ_utilities.time_varying_high_pass_filter(audio2_entering_section,
                                                                             initial_cutoff=400, final_cutoff=1,
                                                                             duration=len(
                                                                                 audio2_entering_section) / 1000)
        # apply the volume fade in to also other frequencies
        audio2_entering_section = audio2_entering_section.fade_in(len(audio2_entering_section))
        audio2 = audio2_entering_section + audio2_without_entering_section

        # Clean up temporary files
        if os.path.exists(adjusted_track1_filename):
            os.remove(adjusted_track1_filename)

        transition_end_instant_relative_to_track2_ms = len(audio2_entering_section)

        return audio1, audio2, exact_out_track_exiting_second * 1000, transition_end_instant_relative_to_track2_ms, track2_initial_ms_for_transition * 1000

    def closest(self, lst, secondsTransition, fromBack):
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