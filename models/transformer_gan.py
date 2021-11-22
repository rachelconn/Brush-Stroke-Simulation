import torch
from torch import nn
from models.transformer import Transformer

class TransformerGAN:
    def __init__(self):
        self.generator = Transformer(2, 2).to('cuda')

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0002,
        )

    def train(self, dataset, batch_size=128, num_examples=100_000):
        print_every = 1_000
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
                print(f'Finished sample {i}.\n  Average loss: {total_loss / print_every}')
                total_loss = 0

            if i == num_examples:
                return
