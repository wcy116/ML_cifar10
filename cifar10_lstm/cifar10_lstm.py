import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', default=5, type=int, help='Epoch')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--device', default='cuda',
                    type=str, help='Device to train on')
parser.add_argument('--model_dir', default='saved_models',
                    type=str, help='Dir to save models')
parser.add_argument('--do_train', default=False,
                    action='store_true', help='Do training')
parser.add_argument('--do_test', default=False,
                    action='store_true', help='Do testing')
parser.add_argument('--resume', default=False,
                    action='store_true', help='Resume training')
parser.add_argument('--ckpt_dir', default=None,
                    type=str, help='Initial checkpoint')
args = parser.parse_args()

torch.manual_seed(233)

current_time = time.strftime('%Y_%m_%d_%H_%M_%S')

train_transform = transforms.Compose([
    transforms.RandomCrop([32, 32], 5),
    transforms.RandomVerticalFlip(.5),
    transforms.RandomHorizontalFlip(.5),
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                         [0.24703223, 0.24348513, 0.26158784])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [
                         0.24703223, 0.24348513, 0.26158784])
])


def load_data(val_fold=10):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=train_transform)
    trainsets = torch.utils.data.ConcatDataset([trainset for _ in range(5)])
    splits = [int(len(trainsets) * (1 - 1 / val_fold)),
              int(len(trainsets) * 1 / val_fold)]
    trainsets, valsets = torch.utils.data.dataset.random_split(
        trainsets, splits)

    trainLoader = torch.utils.data.DataLoader(
        trainsets, batch_size=args.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(
        valsets, batch_size=args.batch_size)

    testsets = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=test_transform)
    testLoader = torch.utils.data.DataLoader(
        testsets, batch_size=args.batch_size, shuffle=False)

    print(
        f'# Train set: {len(trainsets)}\n# Val set: {len(valsets)}\n# Test set: {len(testsets)}')
    return trainLoader, valLoader, testLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSTM1 = nn.LSTM(32*3, 512, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.LSTM2 = nn.LSTM(512, 256, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        for name in self.LSTM1.state_dict():
            if name.startswith('weight'):
                nn.init.kaiming_normal_(self.LSTM1.state_dict()[name])
        for name in self.LSTM2.state_dict():
            if name.startswith('weight'):
                nn.init.kaiming_normal_(self.LSTM2.state_dict()[name])

        self.fc = nn.Linear(256, 128)
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        out1, _ = self.LSTM1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.LSTM2(out1)
        out2 = self.dropout2(out2)
        out = F.relu(self.fc(out2))
        return self.output(out[:, -1, :])


def accuracy(dataloader, net, device=args.device):
    pred_hist = []
    for data, label in tqdm(dataloader):
        data = data.to(device)
        label = label.to(device)
        data = data.view(-1, 32, 32*3)
        out = net(data)
        correct = torch.argmax(out, dim=1) == label
        pred_hist += correct.cpu().tolist()
    acc = np.mean(pred_hist)

    return acc


def train(net, Loss, Opt, Sched):
    train_loss = []
    lr = []
    train_acc = []
    val_acc = []
    best_acc = 0
    best_model_wts = None

    start = time.time()
    for e in range(args.epoch):
        print(f'Epoch {e+1}/{args.epoch}')
        net.train()

        for i, (data, label) in enumerate(trainLoader):
            data = data.to(args.device)
            label = label.to(args.device)

            data = data.view(-1, 32, 32*3)
            out = net(data)
            loss = Loss(out, label)

            Opt.zero_grad()
            loss.backward()
            Opt.step()
            if isinstance(Sched, torch.optim.lr_scheduler.CyclicLR):
                Sched.step()  # CyclicLR

            if i % 500 == 0:
                l_r = Opt.state_dict()['param_groups'][0]['lr']
                print(
                    f'step {i:4d}, loss={loss.data.item():.4f}, lr={l_r:.6f}')
                # logging history
                train_loss.append(loss.item())
                lr.append(l_r)

        net.eval()
        acc = accuracy(valLoader, net)
        val_acc.append(acc)

        if isinstance(Sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            Sched.step(train_loss[-1])  # ReduceLROnPlateau

        print(f'Val acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(net.state_dict())
    end = time.time()
    print(f'Train cost {end-start:.4f}s')
    print('\nTesting...')
    net.eval()
    net.load_state_dict(best_model_wts)
    test_acc = accuracy(testLoader, net)
    print(f'Acc = {test_acc}')

    print('Saving model...')
    torch.save({'parameters': net.state_dict(),
                'optimizer': Opt.state_dict()},
                os.path.join(args.model_dir, f'{current_time}.model'))

    print('Saving history...')
    history = {'train_acc': train_acc, 'train_loss': train_loss,
                'val_acc': val_acc, 'lr': lr,
                'test_acc': test_acc}
    with open(os.path.join(args.model_dir, f'{current_time}_history.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    if not args.do_train and not args.do_test and not args.resume:
        raise NotImplementedError(
            'At least do one of them: {train, test, resume}')
    if args.resume and args.ckpt_dir is None:
        raise NotImplementedError('Need to provide ckpt for resuming')

    trainLoader, valLoader, testLoader = load_data()

    net = Net()
    Loss = nn.CrossEntropyLoss()
    Opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    Sched = optim.lr_scheduler.ReduceLROnPlateau(Opt, 'min', factor=0.5, patience=1)

    if args.do_train or args.resume:
        print('*** Train ***')
        if args.resume:
            print('*** Load Model and Resume Training ***')
            try:
                checkpoint = torch.load(args.ckpt_dir)
                net.load_state_dict(checkpoint['parameters'])
                Opt.load_state_dict(checkpoint['optimizer'])
            except:
                net = torch.load(args.ckpt_dir)
        net.to(args.device)

        train(net, Loss, Opt, Sched)

    elif args.do_test:
        if args.ckpt_dir is None:
            raise NotImplementedError('Please provide checkpoint file')
        print('\n*** Load Model ***')
        try:
            checkpoint = torch.load(args.ckpt_dir)
            net.load_state_dict(checkpoint['parameters'])
        except:
            net = torch.load(args.ckpt_dir)
        print('Parameters')
        for name in net.state_dict():
            print(name,'\t', net.state_dict()[name].size())
        print('\n*** Test ***')        
        net.to(args.device)
        print(f'Accuracy = {accuracy(testLoader, net):.2%}')
