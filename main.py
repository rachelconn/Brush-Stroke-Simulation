import numpy as np
from data.data_loader import load_dataset
from models.transformer_gan import TransformerGAN


TRAINING_DATA_CSV_FOLDER = 'training_data'
NPZ_SAVE_FOLDER = 'numpy_training_data'

if __name__ == '__main__':
    dataset = load_dataset(TRAINING_DATA_CSV_FOLDER)

    transformer_gan = TransformerGAN()
    transformer_gan.train(dataset)
