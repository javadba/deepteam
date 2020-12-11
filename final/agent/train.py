from .planner import Planner, save_model
from .planner import spatial_argmax
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import torch.nn.functional as F
import torchvision.transforms as TF
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    """
    Your code here, modify your HW4 code
    """
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'detector.th')))
    loss = torch.nn.CrossEntropyLoss()
    #loss = torch.nn.BCEWithLogitsLoss()
    #loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    import inspect

    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    #train_data = load_data('3xrandomagentx1250z', transform=transform, num_workers=args.num_workers)
    #train_data = load_data('emVsAI1kz', transform=transform, num_workers=args.num_workers)
    #train_data = load_data('2aix2rand2kz', transform=transform, num_workers=args.num_workers)
    train_data = load_data('2x2x1250z', transform=transform, num_workers=args.num_workers)
    #train_data = load_data('oneone2500z', transform=transform, num_workers=args.num_workers)
    #valid_data = load_data('2aix2rand2kz', num_workers=args.num_workers)
    global_step = 0

    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        acc = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            accuracy = (pred.argmax(1) == label).detach().cpu().numpy()
            #accuracy = ((pred>0).long() == label).detach().cpu().numpy()
            acc.extend(accuracy)
            #print(label, pred)
            #loss_val = (loss(pred, label)*pred).mean() / pred.mean()
            loss_val = loss(pred, label)
#            if train_logger is not None:
#                train_logger.add_scalar('loss', loss_val, global_step)
#                train_logger.add_scalar('accuracy', np.mean(acc), global_step)
                #if global_step % 10 == 0:
                    #log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            losses.append(loss_val.detach().cpu().numpy())
        avg_loss = np.mean(losses)
#        model.eval()
#        acc_valid = []
#        for img, label in valid_data:
#            img, label = img.to(device), label.to(device)
#            pred = model(img, apply_sigmoid=False)
#            accuracy = ((pred>0).long() == label).detach().cpu().numpy()
#            acc_valid.extend(accuracy)
#            valid_logger.add_scalar('accuracy', np.mean(acc_valid), global_step)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f' % (epoch, avg_loss, np.mean(acc)))
        sys.stdout.flush()
        save_model(model)
    save_model(model)


def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    first_label = bool(label[0].cpu().detach().numpy())
    first_pred = float(pred[0].cpu().detach().numpy())
    ax.add_artist(plt.Circle(WH2 * ((1,1)), 10, ec='g', fill=first_label, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * ((1,1)), 10, ec='r', fill=True, lw=1.5, alpha=max(0,first_pred)))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/train2')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ToTensor()])')

    args = parser.parse_args()
    train(args)
