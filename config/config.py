from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'name',
])

default_model_params = ModelParams(**{
    'name': 'test',
})

TrainingParams = namedtuple('TrainingParams', [
    'lr',
])

default_training_params = TrainingParams(**{
    'lr': 0.0002,
})
