import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.utils import save_image
import os
import time

batch_size = 32


class SOM(nn.Module):
    def load_Data(self):
        train_data = datasets.MNIST('.', download=True)

        a = train_data.data[0:1000]
        a1 = train_data.targets[0:1000]
        return a, a1

    def __init__(self, input_size, out_size, lr):

        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.lr = lr
        self.sigma = out_size / 2  #
        self.weight = nn.Parameter(torch.randn(input_size, out_size * out_size), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)
        self.locations = nn.Parameter(torch.Tensor(list(self.getindex())), requires_grad=False)

    def getindex(self):
        for x in range(self.out_size):
            for y in range(self.out_size):
                yield (x, y)

    def computeh(self, input, current_sigma):
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)
        dists = self.pdist_fn(input, batch_weight)
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]
        return bmu_locations, losses.sum().div_(batch_size).item(), bmu_indexes
    def trainL(self, input, current_iter, max_iter):
        batch_size = input.size()[0]
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction
        bmu_locations, loss, bmu_index = self.forward(input)
        distance_squares = self.locations.float() - bmu_locations.float()  # as hamesh kamish mi kone
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)
        lr_locations = self.computeh(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)
        # print("1",lr_locations)
        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        delta1 = nn.Parameter(torch.zeros(self.input_size, self.out_size * self.out_size), requires_grad=False)
        for i in range(len(delta)):
            delta1[i][int(bmu_index)] = delta[i][int(bmu_index)]
        self.weight.data.add_(delta1)
        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):

        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size * self.out_size)

        images = images.permute(3, 0, 1, 2)
        print(images.size())

        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size)

    def save_f(self):
        torch.save(self.state_dict(), './weightS.pth')

    def load_f(self):
        self.load_state_dict(torch.load('./weightS.pth'))


def test1(train):
    total_epoch = 100
    row = 25
    RES_DIR = "result"

    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    som_instance = SOM(input_size=28 * 28, out_size=row, lr=0.3)
    a, a1 = som_instance.load_Data()

    if train is True:

        for epoch in range(total_epoch):
            running_loss = 0
            start_time = time.time()
            for (X, Y) in zip(a, a1):
                # print(X,Y)
                X = X.view(-1, 28 * 28 * 1)
                loss = som_instance.trainL(X, epoch, total_epoch)
                running_loss += loss

            print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss, time.time() - start_time))

            if epoch % 5 == 0:
                som_instance.save_result('%s/som_epoch_%d.png' % (RES_DIR, epoch), (1, 28, 28))
                som_instance.save_f()

        som_instance.save_f()

        som_instance.save_result('%s/som_result.png' % (RES_DIR), (1, 28, 28))

    if train is False:
        if os.path.exists('./weightS.pth'):
            som_instance.load_f()

        running_loss = 0
        start_time = time.time()
        clusters = {}
        clusters2 = {}
        for i in range(625):
            clusters[i] = 0;

        for i in range(625):
            clusters2[i] = {};
            for l in range(10):
                clusters2[i][l] = 0

        print(som_instance.weight)
        for (X, Y) in zip(a, a1):
            X = X.view(-1, 28 * 28 * 1)
            bmu_locations, loss, bmu_index = som_instance.forward(X)

            # print(int(bmu_index[0][0]))
            index = int(bmu_index[0][0])
            clusters[index] += 1
            # print(Y)
            clusters2[index][int(Y)] += 1
            running_loss += loss
        print(clusters2)
        clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: item[1], reverse=True)}
        print(clusters)

        print('loss = %.2f, time = %.2fs' % (running_loss, time.time() - start_time))


if __name__ == '__main__':
    # test1(False)
    test1(True)

