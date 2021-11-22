import os
import numpy as np
import tensorflow as tf

MAX_SEQ_LENGTH = 300

def _csv_to_array(text):
    return tf.reshape(tf.strings.to_number(tf.strings.split(text, ',')), (-1, 2))

def _process_entry(line):
    """
        Processes a single line from the dataset into the target input:
        a tuple containing numpy arrays of shape [num_strokes, 2] for the full + simplified paths
    """
    split = tf.strings.split(line, ';')
    full = _csv_to_array(split[0])
    simplified = _csv_to_array(split[1])
    return simplified, full

def load_dataset(folder):
    training_files = [os.path.join(folder, f) for f in os.listdir(folder)]

    dataset = tf.data.TextLineDataset(training_files)
    dataset = dataset.map(_process_entry)
    return dataset
