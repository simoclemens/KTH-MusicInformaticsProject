import math, array

ARRAY_RANGES = {
  8: (-0x80, 0x7f),
  16: (-0x8000, 0x7fff),
  32: (-0x80000000, 0x7fffffff),
} 

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

def get_min_max_value(bit_depth):
  return ARRAY_RANGES[bit_depth]