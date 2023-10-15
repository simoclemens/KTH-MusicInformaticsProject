import numpy as np
import librosa
import librosa.display
#from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label


def determKeyExtraction(samples, sr, second_key_flag=False):

    # compute chromograph
    chromograph = librosa.feature.chroma_cqt(y=samples, sr=sr, bins_per_octave=24)

    # chroma_vals is the amount of each pitch class present in this time interval
    chroma_vals = []
    for i in range(12):
        chroma_vals.append(np.sum(chromograph[i]))
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # dictionary relating pitch names to the associated intensity in the song
    keyfreqs = {pitches[i]: chroma_vals[i] for i in range(12)}

    keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

    # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
    # data above to typical profiles of major and minor keys:
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # finds correlations between the amount of each pitch class in the time interval and the above profiles,
    # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
    min_key_corrs = []
    maj_key_corrs = []
    for i in range(12):
        key_test = [keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
        # correlation coefficients (strengths of correlation for each key)
        maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
        min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

    # names of all major and minor keys
    key_dict = {**{keys[i]: maj_key_corrs[i] for i in range(12)},
                **{keys[i + 12]: min_key_corrs[i] for i in range(12)}}

    # this attribute represents the key determined by the algorithm
    key = max(key_dict, key=key_dict.get)
    best_corr = max(key_dict.values())
    if not second_key_flag:
        return key
    else:
        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        second_key = None
        second_best_corr = None

        for key, corr in key_dict.items():
            if corr > best_corr * 0.9 and corr != best_corr:
                if second_best_corr is None or corr > second_best_corr:
                    second_key = key
                    second_best_corr = corr
        return key, second_key


# def nnKeyExtraction(path):
#     proc = CNNKeyRecognitionProcessor()
#     pred = proc(path)
#     key = key_prediction_to_label(pred)
#     return key

