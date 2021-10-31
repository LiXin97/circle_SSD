from src.net import *
from src.data import MyDataset
from src.utils.utils import *

import tqdm

data_dir = 'data/circle_train'
model_save = 'model/0.pth'
batch_size = 8
device = try_gpu()

def train():
    train_iter = torch.utils.data.DataLoader(MyDataset(data_dir, is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(MyDataset(data_dir, is_train=False),
                                           batch_size)

    net = TinyCircleSSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    num_epochs, timer = 20, Timer()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['class error', 'bbox mae'])
    net = net.to(device)

    for epoch in tqdm.tqdm(range(num_epochs)):
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)

            anchors, cls_preds, bbox_preds = net(X)

            bbox_labels, bbox_masks, cls_labels = multicircle_target(anchors, Y)

            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        print("cls_err = ", cls_err, "bbox_mae = ", bbox_mae)
        # animator.add(epoch + 1, (cls_err, bbox_mae))

    print(f'class err {cls_err:.2e}, bcircle mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
          f'{str(device)}')

    torch.save( net, model_save )


if __name__ == '__main__':
    train()