from .planner import Planner, save_model
from .planner import spatial_argmax
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import torch.nn.functional as F
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Planner().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = torch.nn.BCEWithLogitsLoss()

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('2x2x1250z', transform=transform, num_workers=4)        
    global_step = 0
    for epoch in range(args.num_epoch):
        acc = []
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            accuracy = ((logit>0).long() == label).detach().cpu().numpy()
            acc.extend(accuracy)    
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            losses.append(loss_val.detach().cpu().numpy())
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('accuracy', np.mean(acc), global_step)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f' % (epoch, np.mean(losses), np.mean(acc)))
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
    ax.add_artist(plt.Circle(((1.25,1)), 5, ec='g', fill=first_label, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * ((.75,1)), 5, ec='r', fill=True, lw=1.5, alpha=max(0,first_pred)))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')
    args = parser.parse_args()
    train(args)