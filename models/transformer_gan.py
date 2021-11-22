import os
import torch
from torch import nn
from models.transformer import Transformer

class TransformerGAN:
    def __init__(self, model_params, training_params):
        self.model_params = model_params
        self.training_params = training_params

        self.checkpoint_path = os.path.join('trained', self.model_params.name, 'model.ckpt')

        self.generator = Transformer(2, 2).to('cuda')

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.model_params.lr,
        )

    def train(self, dataset, batch_size=128, num_examples=100_000):
        print_every = 1_000
        save_every = 10_000
        total_loss = 0

        print('Beginning training...')
        dataset = dataset.shuffle(buffer_size=10_000)
        # TODO: use src_key_padding_mask to allow batch sizes > 1
        for i, sample in enumerate(dataset, 1):
            x, y = sample
            x = torch.unsqueeze(torch.from_numpy(x.numpy()), 1).to('cuda')
            y = torch.unsqueeze(torch.from_numpy(y.numpy()), 1).to('cuda')

            pred = self.generator(x, y)
            loss = self.loss(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss
            if i % print_every == 0:
                print(y)
                print(pred)
                print(f'Finished sample {i}.\n  Average loss: {total_loss / print_every}')
                total_loss = 0

            if i % save_every == 0:
                self.save()

            if i == num_examples:
                return

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.generator.state_dict(), self.checkpoint_path)
        print(f'Model saved to {self.checkpoint_path}.')

    def load(self):
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.generator.load_state_dict(ckpt, strict=True)
        print(f'Loaded parameters from {self.checkpoint_path}.')
