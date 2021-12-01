import nvidia_smi
import gc
import os
import torch
from torch import nn
from models.transformer import Transformer

class TransformerGAN:
    def __init__(self, model_params, training_params):
        self.model_params = model_params
        self.training_params = training_params

        self.checkpoint_path = os.path.join('trained', self.model_params.name, 'model.ckpt')

        self.generator = Transformer(2, 64, 2).to('cuda')
        self.load()

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.training_params.lr,
        )

    def train(self, dataset, num_batches=3_000):
        print_every = 300
        save_every = 300
        total_loss = 0

        print('Beginning training...')
        dataset = dataset.shuffle(buffer_size=100).batch(self.training_params.batch_size).repeat()
        # TODO: use src_key_padding_mask to allow batch sizes > 1
        for batch_num, batch in enumerate(dataset, 1):
            x, y = batch
            x /= 255
            y /= 255
            loss = 0
            for sample in range(self.training_params.batch_size):
                sample_x = torch.unsqueeze(torch.from_numpy(x[sample].numpy()), 1).to('cuda')
                sample_y = torch.unsqueeze(torch.from_numpy(y[sample].numpy()), 1).to('cuda')

                sample_pred = self.generator(sample_x, sample_y)
                loss += self.loss(sample_pred, sample_y)

            loss /= self.training_params.batch_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss
            if batch_num % print_every == 0:
                print(x[-1])
                print(y[-1])
                print(sample_pred)
                print(f'Finished batch {batch_num}.\n  Average loss: {total_loss / print_every}')
                total_loss = 0

            if batch_num % save_every == 0:
                self.save()

            if batch_num == num_batches:
                return

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.generator.state_dict(), self.checkpoint_path)
        print(f'Model saved to {self.checkpoint_path}.')

    def load(self):
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.generator.load_state_dict(ckpt, strict=True)
            print(f'Loaded parameters from {self.checkpoint_path}.')
        except:
            print(f'No saved model found in {self.checkpoint_path}, creating new model.')
        finally:
            return
