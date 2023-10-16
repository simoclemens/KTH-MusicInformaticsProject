import math, array
from tracks import TrackFeatures
import os, array, math
from pydub import AudioSegment
import ffmpy


ARRAY_RANGES = {
    8: (-0x80, 0x7f),
    16: (-0x8000, 0x7fff),
    32: (-0x80000000, 0x7fffffff),
}


def low_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where higher frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) above this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = dt / (RC + dt)

    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)

    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            last_val[j] = last_val[j] + (alpha * (original[offset] - last_val[j]))
            filteredArray[offset] = int(last_val[j])

    return seg._spawn(data=filteredArray)


def high_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where lower frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) below this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = RC / (RC + dt)

    minval, maxval = get_min_max_value(seg.sample_width * 8)

    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)

    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i - 1) * seg.channels) + j

            last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
            filteredArray[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filteredArray)


def time_varying_high_pass_filter(seg, initial_cutoff, final_cutoff, duration):
    dt = 1.0 / seg.frame_rate
    isToDecrement = False
    if initial_cutoff > final_cutoff:
        isToDecrement = True
    minval, maxval = get_min_max_value(seg.sample_width * 8)
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    frame_count = int(seg.frame_count())
    last_val = [0] * seg.channels
    cutoff_delta = abs(initial_cutoff - final_cutoff) / (duration * seg.frame_rate)
    cutoff = initial_cutoff

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i - 1) * seg.channels) + j
            if isToDecrement:
                cutoff -= cutoff_delta
                cutoff = max(final_cutoff, cutoff)
            else:
                cutoff += cutoff_delta
                cutoff = min(final_cutoff, cutoff)

            RC = 1.0 / (cutoff * 2 * math.pi)
            alpha = RC / (RC + dt)
            if isToDecrement:
                if cutoff > final_cutoff:
                    last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
                else:
                    last_val[j] = original[offset]
            else:
                last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])

            filteredArray[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filteredArray)


def change_tempo(input_track, target_tempo):
    initial_tempo = input_track.bpm
    playback_speed = target_tempo / initial_tempo

    # Define temporary files
    temp_in = "temp_in_change_tempo.wav"
    temp_out = "temp_out_change_tempo.wav"

    # Load and export the entire audio segment to a temporary file
    audio = AudioSegment.from_file(input_track.file_name)
    audio.export(temp_in, format="wav")

    # Apply speed change using ffmpy
    ff = ffmpy.FFmpeg(
        inputs={temp_in: None},
        outputs={temp_out: '-filter:a "atempo={}"'.format(playback_speed)},
        global_options='-loglevel quiet -y'  # Force overwrite
    )
    ff.run()

    # Load the modified audio segment
    modified_audio = AudioSegment.from_wav(temp_out)

    # Clean up temporary files
    os.remove(temp_in)
    os.remove(temp_out)

    return modified_audio


def gradual_tempo_change(input_track, final_tempo, final_second_for_tempo_increase, num_subsegments=100):
    initial_tempo = input_track.bpm
    audio = AudioSegment.from_file(input_track.file_name)

    tempo_increment_per_subsegment = (final_tempo - initial_tempo) / num_subsegments
    subsegment_duration = final_second_for_tempo_increase * 1000 / num_subsegments
    tempo_segments = []

    current_tempo = initial_tempo
    current_segment_start_time = 0

    for _ in range(num_subsegments):
        current_segment_end_time = current_segment_start_time + subsegment_duration
        subsegment = audio[current_segment_start_time:int(current_segment_end_time)]  # Ensure slicing is precise

        # Calculate playback speed
        playback_speed = current_tempo / initial_tempo

        # Export subsegment to a temporary file
        temp_in = "temp_in.wav"
        temp_out = "temp_out.wav"
        subsegment.export(temp_in, format="wav")

        # Speed up using ffmpy
        ff = ffmpy.FFmpeg(
            inputs={temp_in: None},
            outputs={temp_out: '-filter:a "atempo={}"'.format(playback_speed)},
            global_options='-loglevel quiet -y'  # Force overwrite
        )
        ff.run()

        # Reload the sped-up segment and append
        sped_up_segment = AudioSegment.from_wav(temp_out)
        tempo_segments.append(sped_up_segment)

        current_tempo += tempo_increment_per_subsegment
        current_segment_start_time = current_segment_end_time

        # Clean up temporary files immediately
        os.remove(temp_in)
        os.remove(temp_out)

    # Handle the rest of the track
    segment_after_final_segment = audio[int(final_second_for_tempo_increase * 1000):]  # Ensure precise slicing
    playback_speed_final = current_tempo / initial_tempo
    segment_after_final_segment.export(temp_in, format="wav")

    ff = ffmpy.FFmpeg(
        inputs={temp_in: None},
        outputs={temp_out: '-filter:a "atempo={}"'.format(playback_speed_final)},
        global_options='-loglevel quiet -y'  # Force overwrite
    )
    ff.run()

    sped_up_final = AudioSegment.from_wav(temp_out)
    final_segment_lenth_ms = len(sped_up_final)
    tempo_segments.append(sped_up_final)

    # Clean up temporary files
    os.remove(temp_in)
    os.remove(temp_out)

    gradual_tempo_track = sum(tempo_segments)

    gradual_tempo_track = high_pass_filter(gradual_tempo_track, 100)

    return gradual_tempo_track, final_segment_lenth_ms


def get_min_max_value(bit_depth):
    return ARRAY_RANGES[bit_depth]
