

import pypianoroll
import numpy as np


def metrics(mid_path, verbose=True):
    multitrack = pypianoroll.read(mid_path)
    tracks = multitrack.tracks
    empty_beat_rate = np.array([0] * 4)
    qualified_note_rate = np.array([0] * 4)
    n_pitch_class_used = np.array([0] * 4)
    TD = np.array([[0] * 4] * 4)

    if verbose:
        print("Multitrack information: \n")
        print("========================\n")
        print(multitrack)
        print("========================\n")
        print("Track information: \n")
        print("========================\n")
        print(tracks)

    # np.set_printoptions(threshold=np.inf)
    for i in range(len(tracks)):
        curr_pianoroll = tracks[i].pianoroll

        empty_beat_rate[i] = pypianoroll.metrics.empty_beat_rate(curr_pianoroll, 24)
        # qualified_note_rate[i] = pypianoroll.qualified_note_rate(curr_pianoroll, threshold=3)
        n_pitch_class_used[0] = pypianoroll.n_pitch_classes_used(curr_pianoroll)

        for j in range(len(tracks)):
            TD[i][j] = pypianoroll.tonal_distance(pianoroll_1=tracks[i].pianoroll, pianoroll_2=tracks[j].pianoroll,
                                                  resolution=24) # resolution=24

    return {'empty_beat_rate': empty_beat_rate, 'qualified_note_rate': qualified_note_rate,
            'n_pitch_class_used':n_pitch_class_used, 'TD':TD}


# result = metrics("myexample.midi", verbose=False)
# print(result)
