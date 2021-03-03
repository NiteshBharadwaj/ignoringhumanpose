from pathlib import Path
import random
import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import set_random_seed, disable_grads, enable_grads, evaluate
from utils.data_transform import transform
from dalib.vision.datasets import Office31, OfficeHome
from model.resnet import Resnet, param_lr, getParam, getOptim
from engine.meta_train import meta_train
import higher

domainIdxDict = {'Ar': 0, 'Cl': 1, 'Pr': 2, 'Rw': 3, 'A': 0, 'D': 1, 'W': 2}


def argument_parser():
    parser = argparse.ArgumentParser(
        description='regularize the target by the source')
    parser.add_argument('--verbose',
                        help='Print log file',
                        action='store_true')
    parser.add_argument('--gpu', type=int, help='GPU idx to run', default=0)
    parser.add_argument('--save_dir',
                        type=str,
                        help='Write directory',
                        default='output')
    parser.add_argument('--model',
                        type=str,
                        choices=['resnet'],
                        help='Model',
                        default='resnet')
    parser.add_argument('--source_domain', type=str, default="Cl")
    parser.add_argument('--target_domain', type=str, default="Ar")
    parser.add_argument('--features_lr',
                        type=float,
                        help='Feature extractor learning rate',
                        default=1e-4)
    parser.add_argument('--classifier_lr',
                        type=float,
                        help='Classifier learning rate',
                        default=1e-3)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size',
                        default=64)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='Weight Decay',
                        default=5e-4)
    parser.add_argument('--wandb',
                        type=str,
                        help='Plot on wandb ',
                        default=None)
    parser.add_argument('--random_seed',
                        type=int,
                        help='random seed',
                        default=42)
    parser.add_argument('--dataset',
                        type=str,
                        choices=['office31', 'visda', 'officehome'],
                        help='Dataset',
                        default='officehome')
    parser.add_argument('--data_dir',
                        type=str,
                        metavar='PATH',
                        default=os.path.join('temp', 'data'))
    parser.add_argument('--lam', type=float, help='lambda', default=7e-3)
    parser.add_argument('--gamma', type=float, help='gamma', default=1)
    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='patience for scheduler')
    parser.add_argument('--factor',
                        type=float,
                        default=0.1,
                        help='factor of scheduler')
    parser.add_argument('--meta_loop', type=int, default=1)
    parser.add_argument('--step_size',
                        type=int,
                        default=40,
                        help='step size for scheduler')
    parser.add_argument('--baseline1',
                        action='store_true',
                        default=False,
                        help="train on tgt")
    parser.add_argument('--baseline2',
                        action='store_true',
                        default=False,
                        help="train on src and tgt")
    parser.add_argument('--ours1',
                        action='store_true',
                        default=False,
                        help="train on reweighted src and tgt")
    parser.add_argument('--baseline3',
                        action='store_true',
                        default=False,
                        help="train on tgt with regularization from src")
    parser.add_argument(
        '--baseline4',
        action='store_true',
        default=False,
        help="train on src and tgt with regularization from src")
    parser.add_argument(
        '--ours2',
        action='store_true',
        default=False,
        help='train on tgt with regularization from reweighted src')
    parser.add_argument(
        '--ours3',
        action='store_true',
        default=False,
        help='train on reweighted src and tgt with regularization from  src')
    parser.add_argument(
        '--ours4',
        action='store_true',
        default=False,
        help='train on src and tgt with regularization from reweighted src')
    parser.add_argument(
        '--ours5',
        action='store_true',
        default=False,
        help=
        'train on reweighted src and tgt with regularization from reweighted src'
    )
    return parser


def main(args):
    print('args: ', args)
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.save_dir)
    else:
        wandb = None

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _save_dir = 'results/' + args.save_dir
    Path(_save_dir).mkdir(parents=True, exist_ok=True)
    if not args.verbose:
        sys.stdout = open(f'{_save_dir}/log.txt', 'w')
    print('args: ', args)

    # Create dataloaders
    train_transform = transform(train=True)
    test_transform = transform(train=False)
    data_root = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'office31':
        getDataset = Office31
    elif args.dataset == 'officehome':
        getDataset = OfficeHome
    train_source_dataset = getDataset(root=data_root,
                                      task=args.source_domain + '_train',
                                      download=True,
                                      transform=train_transform)

    print(f'train target_task: {args.target_domain}')
    train_target_dataset = getDataset(root=data_root,
                                      task=args.target_domain + '_train',
                                      download=True,
                                      transform=train_transform)
    train_dataset = train_target_dataset
    if args.baseline4 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.ours1:
        train_target_source_dataset = torch.utils.data.ConcatDataset(
            [train_target_dataset, train_source_dataset])
        train_dataset = train_target_source_dataset
    valid_target_dataset = getDataset(root=data_root,
                                      task=args.target_domain + '_val',
                                      download=True,
                                      transform=test_transform)
    test_target_dataset = getDataset(root=data_root,
                                     task=args.target_domain + '_test',
                                     download=True,
                                     transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=6,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_target_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=6,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_target_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=6,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False)
    train_source_loader = torch.utils.data.DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    if args.model == 'resnet':
        print('Using resnet')
        model_tgt = Resnet(
            num_classes=train_target_dataset.num_classes).to(device)

        optimizer_tgt = getOptim(model_tgt, args)
        scheduler_tgt = torch.optim.lr_scheduler.StepLR(
            optimizer_tgt, step_size=args.step_size, gamma=1e-1)

        model_src = Resnet(
            num_classes=train_source_dataset.num_classes).to(device)
        optimizer_src = getOptim(model_src, args)
        scheduler_src = torch.optim.lr_scheduler.StepLR(
            optimizer_src, step_size=args.step_size, gamma=1e-1)

    for epoch in range(0, args.num_epochs):
        train_valid_queue = iter(valid_loader)
        train_src_queue = iter(train_source_loader)

        model_src.train()
        model_tgt.train()
        for i, (data_train, target_train,
                domain_idx) in enumerate(train_loader):
            ##############get the validate target data ###############################
            try:
                data_valid, target_valid, _ = next(train_valid_queue)
            except StopIteration:
                train_valid_queue = iter(valid_loader)
                data_valid, target_valid, _ = next(train_valid_queue)
            x_tgt_val, y_tgt_val = data_valid.to(device), target_valid.to(
                device)
            ###########get the images of target and source by using index##############
            if args.baseline4 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.ours1:
                print(f'batch={i}')
                source_idx = (domain_idx == domainIdxDict[args.source_domain]
                              ).nonzero().squeeze()
                target_idx = (domain_idx == domainIdxDict[args.target_domain]
                              ).nonzero().squeeze()
                x_tgt = torch.index_select(data_train, 0, target_idx)
                y_tgt = torch.index_select(target_train, 0, target_idx)
                x_src = torch.index_select(data_train, 0, source_idx)
                y_src = torch.index_select(target_train, 0, source_idx)
                x_src, y_src = x_src.to(device), y_src.to(device)
                x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)
            else:
                x_tgt, y_tgt = data_train.to(device), target_train.to(device)
                try:
                    data_train_src, target_train_src, _ = next(train_src_queue)
                except StopIteration:
                    train_src_queue = iter(train_source_loader)
                    data_train_src, target_train_src, _ = next(train_src_queue)
                x_src, y_src = data_train_src.to(device), target_train_src.to(
                    device)
            #############################################################################
            print('############Starting meta learning###############')
            model_tgt_backup = model_tgt.state_dict()
            optimizer_tgt_backup = optimizer_tgt.state_dict()
            model_src_backup = model_src.state_dict()
            optimizer_src_backup = optimizer_src.state_dict()
            w = meta_train(args, model_tgt, model_src, x_tgt, y_tgt, x_src,
                           y_src, x_tgt_val, y_tgt_val, optimizer_tgt,
                           optimizer_src, device)
            if args.ours2:
                A = w
            elif args.ours3 or args.ours1:
                B = w
            elif args.ours4:
                A = w
            elif args.ours5:
                A, B = w
            model_tgt.load_state_dict(model_tgt_backup)
            optimizer_tgt.load_state_dict(optimizer_tgt_backup)
            model_src.load_state_dict(model_src_backup)
            optimizer_src.load_state_dict(optimizer_src_backup)
            #######################normal learning#################################
            print('############Starting normal learning###############')
            yhat_src = model_src(x_src)
            loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')
            if args.ours2 or args.ours4 or args.ours5:
                loss_src = torch.mean(A * loss_src)
            else:
                loss_src = torch.mean(loss_src)
            optimizer_src.zero_grad()
            loss_src.backward()
            optimizer_src.step()

            yhat_tgt = model_tgt(x_tgt)
            loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')
            if args.ours1 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.baseline4:
                yhat_src2 = model_tgt(x_src)
                loss_src2 = F.cross_entropy(yhat_src2, y_src, reduction='none')
                if args.baseline4 or args.ours4 or args.baseline2:
                    loss_src2 = loss_src2 * args.gamma
                    final_loss = torch.cat((loss_tgt, loss_src2), dim=0)
                else:
                    loss_src2 = B * loss_src2 * args.gamma
                    final_loss = torch.cat((loss_tgt, loss_src2), dim=0)
                final_loss = torch.mean(final_loss)
            else:
                final_loss = torch.mean(loss_tgt)
            norm_sum = 0
            if args.baseline3 or args.baseline4 or args.ours2 or args.ours3 or args.ours4 or args.ours5:
                for sw, tw in zip(model_src.parameters(),
                                  model_tgt.parameters()):
                    w_diff = tw - sw
                    w_diff_norm = torch.norm(w_diff)
                    norm_sum = norm_sum + w_diff_norm**2
                norm_sum = norm_sum * args.lam
                final_loss += norm_sum
            optimizer_tgt.zero_grad()
            final_loss.backward()
            optimizer_tgt.step()
            if args.wandb is not None:
                wandb.log({"norm": norm_sum})
        print(f'Finished epoch {epoch}')
        print('Starting validation...')
        val_tgt_acc, val_tgt_loss = evaluate(model_tgt, valid_loader, device)
        test_tgt_acc, test_tgt_loss = evaluate(model_tgt, test_loader, device)

        scheduler_tgt.step()
        if args.baseline3 or args.baseline4 or args.ours2 or args.ours3 or args.ours4 or args.ours5:
            scheduler_src.step()
        if args.wandb is not None:
            wandb.log({
                "epoch": epoch,
                "val_tgt_acc": val_tgt_acc,
                "val_tgt_loss": val_tgt_loss,
                "test_tgt_acc_per_epoch": test_tgt_acc,
                "test_tgt_loss_per_epoch": test_tgt_loss
            })

    torch.save(model_tgt.state_dict(), f'{_save_dir}/final_model.pt')
    test_tgt_acc, test_tgt_loss = evaluate(model_tgt, test_loader, device)
    print(f'test tgt acc: {test_tgt_acc}')
    print(f'test tgt loss: {test_tgt_loss}')
    with open(f'{_save_dir}/results.txt', 'a') as res:
        res.write(
            f'pretraining domain:{args.source_domain}, finetuning domain:{args.target_domain} \n test accuracy: {test_tgt_acc}, test loss: {test_tgt_loss}'
        )
    if args.wandb is not None:
        wandb.log({
            "test_tgt_acc": test_tgt_acc,
            "test_tgt_loss": test_tgt_loss
        })


if __name__ == "__main__":
    parser = argument_parser()
    main(parser.parse_args())