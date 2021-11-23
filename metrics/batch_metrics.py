

import pypianoroll
import numpy as np
import os
import metrics


def batch_metrics(folder, verbose=True):
    batch_empty_beat_rate = np.array([0] * 4)
    batch_qualified_note_rate = np.array([0] * 4)
    batch_n_pitch_class_used = np.array([0] * 4)
    batch_TD = np.array([[0] * 4] * 4)
    total_files = 0

    for _, _, files in os.next(folder):
        total_files = len(files)
        for file in files:
            if file.endswith('.midi'):
                mid_path = folder + '/' + file
                each = metrics(mid_path, verbose=False)
                batch_empty_beat_rate += each['empty_beat_rate']
                batch_qualified_note_rate += each['qualified_note_rate']
                batch_n_pitch_class_used += each['n_pitch_class_used']
                batch_TD += each['TD']


    if total_files == 0:
        return

    return {'batch_empty_beat_rate': batch_empty_beat_rate / total_files,
            'batch_qualified_note_rate': batch_qualified_note_rate / total_files,
            'batch_n_pitch_class_used': batch_n_pitch_class_used / total_files,
            'batch_TD': batch_TD / total_files}

