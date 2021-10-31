from src.net import *
from src.data import MyDataset
from src.utils.utils import *

import tqdm
import argparse

# data_dir = 'data/circle_train'
# model_save = 'model/0.pth'
batch_size = 16
device = try_gpu()

def train(data_dir, model_save, num_classes, num_epochs):
    train_iter = torch.utils.data.DataLoader(MyDataset(data_dir, is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(MyDataset(data_dir, is_train=False),
                                           batch_size)

    net = TinyCircleSSD(num_classes=num_classes)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    timer = Timer()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['class error', 'bbox mae'])
    net = net.to(device)
    net.train()

    timer.start()

    for epoch in range(num_epochs):
        loop = tqdm.tqdm(train_iter, total=len(train_iter), leave = True)
        for features, target in loop:
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)

            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multicircle_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=l.mean().item(),
                             cls_err= 1. - cls_eval(cls_preds, cls_labels)/cls_labels.numel(),
                             bcircle_mae=bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.numel())

    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
          f'{str(device)}')

    torch.save( net, model_save )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir",
                        help="path to dataset folder", type=str, required=True)
    parser.add_argument("-m", "--model_save",
                        help="path to model save", type=str, required=True)
    parser.add_argument("-n", "--num_classes",
                        help="num of classes", type=int, default=1)
    parser.add_argument("-e", "--num_epochs",
                        help="num of epochs", type=int, default=20)
    args = parser.parse_args()

    train(args.data_dir, args.model_save, args.num_classes, args.num_epochs)