import numpy as np
import tensorflow as tf
from data.data_loader import load_dataset
from models.transformer_gan import TransformerGAN
from config.config import default_model_params, default_training_params


model_params = default_model_params
training_params = default_training_params

TRAINING_DATA_CSV_FOLDER = training_params.training_data_folder
NPZ_SAVE_FOLDER = 'numpy_training_data'

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        dataset = load_dataset(TRAINING_DATA_CSV_FOLDER)

        transformer_gan = TransformerGAN(model_params, training_params)
        transformer_gan.train(dataset, training_params.num_batches)
