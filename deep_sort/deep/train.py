import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import ReNet

def train(epoch):
    print('\nEpoch : %d'%(epoch + 1))
    renet.train()
    training_loss = 0.0
    train_loss = 0.0
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()

    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = renet(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)
        if (idx + 1) % interval == 0:
            end = time.time()
            print('Training ...')
            print('[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%'.format(
                100.0 * (idx + 1) / len(trainloader), end - start, training_loss / interval,
                correct, total, 100.0 * correct / total
            ))
            training_loss = 0.0
            start = time.time()
    return train_loss / len(trainloader), 1.0 - (correct / total)

def val(epoch):
    global best_acc
    renet.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = renet(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        print('Testing ...')
        end = time.time()
        print('[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%'.format(
            100.0 * (idx + 1) / len(testloader), end - start, test_loss / len(testloader),
            correct, total, 100.0 * correct / total))
    acc = 100.0 * correct / total
    if acc > best_acc:
        best_acc = acc
        print('Saving parameters into the most accurate model')
        checkpoint = {
            'net_dict':renet.state_dict(),
            'acc':acc,
            'epoch':epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(testloader), 1.0 - correct / total

def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)
    x_epoch.append(epoch)

    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('train.jpg')

def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print('Learning rate decayed to {}'.format(lr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on market1501')
    parser.add_argument('--data-dir', default='Market1501/pytorch', type=str)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--lr', default=0.3, type=float)
    parser.add_argument('--interval', '-i', default=20, type=int)
    parser.add_argument('--batch_size', '-b', default=64, type=int)
    parser.add_argument('--resume', '-r', action='store_true')
    args = parser.parse_args()
    print('Parameter setting: ', args)

    #device
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    if torch.cuda.is_available() and not args.no_cuda:
        print('Use CUDA: {}'.format(torch.cuda.is_available()))
        cudnn.benchmark = True
    print('On device: {}'.format(device))

    root = args.data_dir
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((128, 64), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
        batch_size=args.batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(val_dir, transform=transform_val),
        batch_size=args.batch_size, shuffle=True
    )

    num_classes = max(len(trainloader.dataset.classes), len(valloader.dataset.classes))
    print('Number of train classes: ', len(trainloader.dataset.classes))
    print('Number of test classes: ', len(valloader.dataset.classes))
    start_epoch = 0
    renet = ReNet(num_classes=num_classes)
    if args.resume:
        assert os.path.isfile('./checkpoint/ckpt.t7'), "Error: no checkpoint file found!"
        print('Loading from checkpoint/ckpt.t7')
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net_dict = checkpoint['net_dict']
        renet.load_state_dict(net_dict)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('Resume epoch: ', start_epoch)
    renet.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(renet.parameters(), args.lr, momentum=0.8, weight_decay=5e-4)
    best_acc = 0

    x_epoch = []
    record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title='loss')
    ax1 = fig.add_subplot(122, title='top1err')

    for epoch in range(start_epoch, start_epoch + 40):
        print('Traning at epoch {}:'.format(epoch + 1))
        train_loss, train_err = train(epoch)
        print('Validating at epoch {}:'.format(epoch + 1))
        val_loss, val_err = val(epoch)
        draw_curve(epoch, train_loss, train_err, val_loss, val_err)
        if (epoch + 1) % 10 == 0:
            lr_decay()
    
