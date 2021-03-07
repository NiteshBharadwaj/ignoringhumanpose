
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
from pathlib import Path
import sys

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELossNoReduction, JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

from lbi_utils.utils import set_random_seed

from engine.meta_train import meta_train
import dataset
import models

domainIdxDict = {'Ar': 0, 'Cl': 1, 'Pr': 2, 'Rw': 3, 'A': 0, 'D': 1, 'W': 2}


def argument_parser():
    parser = argparse.ArgumentParser(
        description='regularize the target by the source')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
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
    parser.add_argument('--source_domain', type=str, default="COCO")
    parser.add_argument('--target_domain', type=str, default="MPII")
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
                        default=8)
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
                        choices=['coco', 'augmend_coco', 'mpii'],
                        help='Dataset',
                        default='coco')
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
def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def main(args):
    print('args: ', args)
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.save_dir)
    else:
        wandb = None
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    print("*****************************************************************************")
    print("Starting")
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _save_dir = 'results/' + args.save_dir
    Path(_save_dir).mkdir(parents=True, exist_ok=True)
    if not args.verbose:
        sys.stdout = open(f'{_save_dir}/log.txt', 'w')
    print('args: ', args)
    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    gpus = [int(i) for i in config.GPUS.split(',')]
    criterion = JointsMSELossNoReduction(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT, logger=logger
    ).cuda()
    criterion_reduce = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    # Create dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform =  transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform =         transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    data_root = os.path.join(args.data_dir, args.dataset)
    train_source_dataset = eval('dataset.'+config.DATASET_SRC.DATASET)(
        config,
        config.DATASET_SRC.ROOT,
        config.DATASET_SRC.TRAIN_SET,
        True,
        train_transform
    )
    train_target_dataset = eval('dataset.'+config.DATASET_TARGET.DATASET)(
        config,
        config.DATASET_TARGET.ROOT,
        config.DATASET_TARGET.TRAIN_SET,
        True,
        train_transform
    )
    train_dataset = train_target_dataset
    if args.baseline4 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.ours1:
        train_target_source_dataset = torch.utils.data.ConcatDataset(
            [train_target_dataset, train_source_dataset])
        train_dataset = train_target_source_dataset
    valid_target_dataset = eval('dataset.'+config.DATASET_TARGET.DATASET)(
        config,
        config.DATASET_TARGET.ROOT,
        config.DATASET_TARGET.VAL_SET,
        True,
        test_transform
    )
    test_target_dataset = eval('dataset.'+config.DATASET_TARGET.DATASET)(
        config,
        config.DATASET_TARGET.ROOT,
        config.DATASET_TARGET.TEST_SET,
        False,
        test_transform
    )

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
    print("*****************************************************************************")
    #print("MOdel", model_tgt)

    if 1:#args.model == 'resnet':
        logger.info('Using resnet')
        gpus = [int(i) for i in config.GPUS.split(',')]
        logger.info("*****************************************************************************")
        print("Loading target")
        model_tgt =  eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=True
        ).to(device)
        logger.info("*****************************************************************************")
        print("MOdel", model_tgt)
        #model_tgt = torch.nn.DataParallel(model_tgt, device_ids=gpus).cuda()
        optimizer_tgt = get_optimizer(config, model_tgt)
        scheduler_tgt = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_tgt, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
        )

        model_src = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=True
        ).to(device)
        #model_src = torch.nn.DataParallel(model_src, device_ids=gpus).cuda()
        optimizer_src = get_optimizer(config, model_src)
        scheduler_src = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_src, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
        )

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        train_valid_queue = iter(valid_loader)
        train_src_queue = iter(train_source_loader)

        model_src.train()
        model_tgt.train()
        for i, (data_input_train, target_train, target_weight_train,
                meta_dict_train) in enumerate(train_loader):
            ##############get the validate target data ###############################
            try:
                data_input_valid, target_valid, target_weight_valid, _ = next(train_valid_queue)
            except StopIteration:
                train_valid_queue = iter(valid_loader)
                data_input_valid, target_valid, target_weight_valid, _ = next(train_valid_queue)
            x_tgt_val, y_tgt_val, w_tgt_val = data_input_valid.to(device), target_valid.to(
                device), target_weight_valid.to(device)
            ###########get the images of target and source by using index##############
            if args.baseline4 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.ours1:
                pass
                # print(f'batch={i}')
                # source_idx = (domain_idx == domainIdxDict[args.source_domain]
                #               ).nonzero().squeeze()
                # target_idx = (domain_idx == domainIdxDict[args.target_domain]
                #               ).nonzero().squeeze()
                # x_tgt = torch.index_select(data_train, 0, target_idx)
                # y_tgt = torch.index_select(target_train, 0, target_idx)
                # x_src = torch.index_select(data_train, 0, source_idx)
                # y_src = torch.index_select(target_train, 0, source_idx)
                # x_src, y_src = x_src.to(device), y_src.to(device)
                # x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)
            else:
                x_tgt, y_tgt, w_tgt = data_input_train.to(device), target_train.to(device), target_weight_train.to(device)
                try:
                    data_input_src, target_src, target_weight_src, _ = next(train_src_queue)
                except StopIteration:
                    train_src_queue = iter(train_source_loader)
                    data_input_src, target_src, target_weight_src, _ = next(train_src_queue)
                x_src, y_src, w_src = data_input_src.to(device), target_src.to(
                    device), target_weight_src.to(device)
            #############################################################################
            #logger.info('############Starting meta learning###############')
            model_tgt_backup = model_tgt.state_dict()
            optimizer_tgt_backup = optimizer_tgt.state_dict()
            model_src_backup = model_src.state_dict()
            optimizer_src_backup = optimizer_src.state_dict()
            w = meta_train(args, model_tgt, model_src, x_tgt, y_tgt, w_tgt, x_src,
                           y_src, w_src, x_tgt_val, y_tgt_val, w_tgt_val, optimizer_tgt,
                           optimizer_src, device,criterion,logger)
            if args.ours2:
                A = w
            elif args.ours3 or args.ours1:
                B = w
            elif args.ours4:
                A = w
            elif args.ours5:
                A, B = w
            #logger.info(A.shape)
            model_tgt.load_state_dict(model_tgt_backup)
            optimizer_tgt.load_state_dict(optimizer_tgt_backup)
            model_src.load_state_dict(model_src_backup)
            optimizer_src.load_state_dict(optimizer_src_backup)
            #######################normal learning#################################
            #logger.info('############Starting normal learning###############')
            yhat_src = model_src(x_src)
            loss_src = criterion(yhat_src,y_src,w_src) # TODO: Reduction should be none here
            if args.ours2 or args.ours4 or args.ours5:
                loss_src = torch.mean(A * loss_src)
            else:
                loss_src = torch.mean(loss_src)
            optimizer_src.zero_grad()
            loss_src.backward()
            optimizer_src.step()

            yhat_tgt = model_tgt(x_tgt)
            loss_tgt = criterion(yhat_tgt,y_tgt,w_tgt) # TODO: Reduction should be none here
            if args.ours1 or args.ours3 or args.ours4 or args.ours5 or args.baseline2 or args.baseline4:
                pass
                # yhat_src2 = model_tgt(x_src)
                # loss_src2 = F.cross_entropy(yhat_src2, y_src, reduction='none')
                # if args.baseline4 or args.ours4 or args.baseline2:
                #     loss_src2 = loss_src2 * args.gamma
                #     final_loss = torch.cat((loss_tgt, loss_src2), dim=0)
                # else:
                #     loss_src2 = B * loss_src2 * args.gamma
                #     final_loss = torch.cat((loss_tgt, loss_src2), dim=0)
                # final_loss = torch.mean(final_loss)
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
        logger.info(f'Finished epoch {epoch}')
        logger.info('Starting validation...')
        val_tgt_acc, val_tgt_loss = validate(config, valid_loader, valid_loader.dataset, model_tgt,
                                  criterion_reduce, final_output_dir, tb_log_dir,
                                  writer_dict) # TODO
        test_tgt_acc, test_tgt_loss = validate(config, test_loader, test_loader.dataset, model_tgt,
                                  criterion_reduce, final_output_dir, tb_log_dir,
                                  writer_dict) #TODO

        scheduler_tgt.step()
        if args.baseline3 or args.baseline4 or args.ours2 or args.ours3 or args.ours4 or args.ours5:
            scheduler_src.step()
        if args.wandb is not None:
            logger.info({
                "epoch": epoch,
                "val_tgt_acc": val_tgt_acc,
                "val_tgt_loss": val_tgt_loss,
                "test_tgt_acc_per_epoch": test_tgt_acc,
                "test_tgt_loss_per_epoch": test_tgt_loss
            })
            wandb.log({
                "epoch": epoch,
                "val_tgt_acc": val_tgt_acc,
                "val_tgt_loss": val_tgt_loss,
                "test_tgt_acc_per_epoch": test_tgt_acc,
                "test_tgt_loss_per_epoch": test_tgt_loss
            })

    torch.save(model_tgt.state_dict(), f'{_save_dir}/final_model.pt')
    test_tgt_acc, test_tgt_loss = validate(config, test_loader, test_loader.dataset, model_tgt,
                                  criterion_reduce, final_output_dir, tb_log_dir,
                                  writer_dict)  #TODO
    logger.info(f'test tgt acc: {test_tgt_acc}')
    logger.info(f'test tgt loss: {test_tgt_loss}')
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
