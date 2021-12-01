from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'name',
])

default_model_params = ModelParams(**{
    'name': 'test_5_16_batch',
})

TrainingParams = namedtuple('TrainingParams', [
    'lr',
    'batch_size',
    'num_batches',
    'training_data_folder',
])

default_training_params = TrainingParams(**{
    'lr': 0.0000001,
    'batch_size': 8,
    'num_batches': 100_000,
    'training_data_folder': 'training_data_5',
})
