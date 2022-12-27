
from model import FCN
import torch
import os
import sys
from tqdm import tqdm
from torch import nn,optim
from torch.nn import functional as F
from dataloader.dataloader import get_train_loader
from dataloader.RGBXdataset import RGBXDataset
from eval_tool import label_accuracy_score

from config import config

model = 'FCN-TIP'
net = FCN(config.num_classes, config.hyper_parm)

result_path = './result_{}.txt'.format(model)
if os.path.exists(result_path):
    os.remove(result_path)

optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

if torch.cuda.is_available():
    torch.cuda.set_device(config.GPU_ID)
    net.cuda()
    criterion = criterion.cuda()

    # data loader
    train_loader, train_sampler = get_train_loader(RGBXDataset)


def train():
    for e in range(config.epoch):
        net.train()
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        for idx in pbar:
            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda()
            modal_xs = modal_xs.cuda()
            batchdata = [imgs, modal_xs]
            batchlabel = gts.cuda()

            output = net(batchdata)
            output = F.log_softmax(output, dim=1)
            loss = criterion(output, batchlabel)

            pred = output.argmax(dim=1).squeeze().data.cpu()
            real = batchlabel.data.cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item() * batchlabel.size(0)
            label_true = torch.cat((label_true, real), dim=0)
            label_pred = torch.cat((label_pred, pred), dim=0)

        train_loss /= len(pbar)
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label_true.numpy(), label_pred.numpy(), config.num_classes)

        print('\nepoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e+1, train_loss, acc, acc_cls, mean_iu, fwavacc))

        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
                e+1, train_loss, acc, acc_cls, mean_iu, fwavacc))

        # save model every 10 epochs, Can be modified to save the best model in the validation set
        if (e+1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(config.save_dir, 'train_model_{}.pth'.format(e+1)))


if __name__ == "__main__":

    train()