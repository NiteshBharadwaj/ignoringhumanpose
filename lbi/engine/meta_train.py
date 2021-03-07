from pathlib import Path
import random
import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import higher


def meta_train(args, model_tgt, model_src, x_tgt, y_tgt, w_tgt, x_src, y_src, w_src,
               x_tgt_val, y_tgt_val, w_tgt_val, optimizer_tgt, optimizer_src, device, criterion, logger):
    meta_train_mode = None
    if args.baseline2 or args.ours1:
        meta_train_mode = meta_train_ours1
    elif args.ours2 or args.baseline1 or args.baseline3:
        meta_train_mode = meta_train_ours2 # Here
    elif args.ours3 or args.baseline4:
        meta_train_mode = meta_train_ours3
    elif args.ours4:
        meta_train_mode = meta_train_ours4
    elif args.ours5:
        meta_train_mode = meta_train_ours5
    else:
        meta_train_mode = meta_train_ours2 # Here
    return meta_train_mode(args, model_tgt, model_src, x_tgt, y_tgt,w_tgt, x_src,
                           y_src,w_src, x_tgt_val, y_tgt_val,w_tgt_val, optimizer_tgt,
                           optimizer_src, device, criterion, logger)


def meta_train_ours1(args, model_tgt, model_src, x_tgt, y_tgt, x_src, y_src,
                     x_tgt_val, y_tgt_val, optimizer_tgt, optimizer_src,
                     device):
    model_tgt.train()
    eps_B = None
    for _ in range(args.meta_loop):
        with higher.innerloop_ctx(model_tgt,
                                  optimizer_tgt) as (fmodel_tgt,
                                                     foptimizer_tgt):
            yhat_tgt = fmodel_tgt(x_tgt)
            loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')
            
            yhat_src = fmodel_tgt(x_src)
            loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')

            if eps_B is None:
                eps_B = torch.zeros(loss_src.size(0),
                                    requires_grad=True).to(device)
            else:
                eps_B.requires_grad = True
            loss_src = eps_B * loss_src * args.gamma
            final_loss = torch.cat((loss_tgt, loss_src), dim=0)
            final_loss = torch.mean(final_loss)
            foptimizer_tgt.step(final_loss)
            yhat_tgt_val = fmodel_tgt(x_tgt_val)
            loss_tgt_val = F.cross_entropy(yhat_tgt_val, y_tgt_val)
            eps_B_grads = torch.autograd.grad(loss_tgt_val, eps_B)[0].detach()
        B_tilde = torch.clamp(-eps_B_grads, min=0)
        l1_norm = torch.sum(B_tilde)
        if l1_norm != 0:
            B_tilde_min = B_tilde.min()
            B_tilde_max = B_tilde.max()
            B = (B_tilde - B_tilde_min) / (B_tilde_max - B_tilde_min)
        else:
            B = B_tilde
        print(f"B: {B}")
        eps_B = B
    return eps_B


def meta_train_ours2(args, model_tgt, model_src, x_tgt, y_tgt, w_tgt, x_src, y_src, w_src,
                     x_tgt_val, y_tgt_val, w_tgt_val, optimizer_tgt, optimizer_src,
                     device, criterion, logger):
    model_tgt.train()
    model_src.train()
    eps_A = None
    for _ in range(args.meta_loop):
        with higher.innerloop_ctx(model_src,
                                  optimizer_src) as (fmodel_src,
                                                     foptimizer_src):
            yhat_src = fmodel_src(x_src)
            loss_src = criterion(yhat_src, y_src, w_src)
            #logger.info(loss_src.shape)
            #logger.info(f"loss_src without weight={loss_src.mean().item():.2f}")
            if eps_A is None:
                eps_A = torch.zeros(loss_src.size(),
                                    requires_grad=True).to(device)
            else:
                eps_A.requires_grad = True
            loss_src = torch.mean(eps_A * loss_src)
            #logger.info(f'loss_src weight={loss_src.item():.2f}')
            foptimizer_src.step(loss_src)
            with higher.innerloop_ctx(model_tgt,
                                      optimizer_tgt) as (fmodel_tgt,
                                                         foptimizer_tgt):
                yhat_tgt = fmodel_tgt(x_tgt)
                loss_tgt = criterion(yhat_tgt, y_tgt,w_tgt).mean()
                norm_sum = 0
                for sw, tw in zip(fmodel_src.parameters(),
                                  fmodel_tgt.parameters()):
                    w_diff = tw - sw
                    w_diff_norm = torch.norm(w_diff)
                    norm_sum = norm_sum + w_diff_norm**2
                norm_sum = norm_sum * args.lam
                loss_tgt = loss_tgt + norm_sum
                foptimizer_tgt.step(loss_tgt)

                yhat_tgt_val = fmodel_tgt(x_tgt_val)
                loss_tgt_val = criterion(yhat_tgt_val, y_tgt_val,w_tgt_val).mean()
                eps_A_grads = torch.autograd.grad(loss_tgt_val,
                                                  eps_A)[0].detach()
        A_tilde = torch.clamp(-eps_A_grads, min=0)
        l1_norm = torch.sum(A_tilde)
        if l1_norm != 0:
            A_tilde_min = A_tilde.min()
            A_tilde_max = A_tilde.max()
            A = (A_tilde - A_tilde_min) / (A_tilde_max - A_tilde_min)
        else:
            A = A_tilde
        #logger.info(f'A: {A}')
        eps_A = A
    return eps_A


def meta_train_ours3(args, model_tgt, model_src, x_tgt, y_tgt, x_src, y_src,
                     x_tgt_val, y_tgt_val, optimizer_tgt, optimizer_src,
                     device):
    model_tgt.train()
    model_src.train()
    eps_B = None
    for _ in range(args.meta_loop):
        with higher.innerloop_ctx(model_src,
                                  optimizer_src) as (fmodel_src,
                                                     foptimizer_src):
            yhat_src = fmodel_src(x_src)
            loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')
            loss_src = torch.mean(loss_src)
            foptimizer_src.step(loss_src)
            with higher.innerloop_ctx(model_tgt,
                                      optimizer_tgt) as (fmodel_tgt,
                                                         foptimizer_tgt):
                yhat_tgt = fmodel_tgt(x_tgt)
                loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')

                yhat_src = fmodel_tgt(x_src)
                loss_src_2 = F.cross_entropy(yhat_src, y_src, reduction='none')

                if eps_B is None:
                    eps_B = torch.zeros(loss_src_2.size(0),
                                        requires_grad=True).to(device)
                else:
                    eps_B.requires_grad = True
                loss_src_2 = eps_B * loss_src_2 * args.gamma
                final_loss = torch.cat((loss_tgt, loss_src_2), dim=0)
                final_loss = torch.mean(final_loss)
                norm_sum = 0
                for sw, tw in zip(fmodel_src.parameters(),
                                  fmodel_tgt.parameters()):
                    w_diff = tw - sw
                    w_diff_norm = torch.norm(w_diff)
                    norm_sum = norm_sum + w_diff_norm**2
                norm_sum = norm_sum * args.lam
                final_loss += norm_sum
                foptimizer_tgt.step(final_loss)
                yhat_tgt_val = fmodel_tgt(x_tgt_val)
                loss_tgt_val = F.cross_entropy(yhat_tgt_val, y_tgt_val)
                eps_B_grads = torch.autograd.grad(loss_tgt_val,
                                                  eps_B)[0].detach()
        B_tilde = torch.clamp(-eps_B_grads, min=0)
        l1_norm = torch.sum(B_tilde)
        if l1_norm != 0:
            B_tilde_min = B_tilde.min()
            B_tilde_max = B_tilde.max()
            B = (B_tilde - B_tilde_min) / (B_tilde_max - B_tilde_min)
        else:
            B = B_tilde
        print(f"B: {B}")
        eps_B = B
    return eps_B


def meta_train_ours4(args, model_tgt, model_src, x_tgt, y_tgt, x_src, y_src,
                     x_tgt_val, y_tgt_val, optimizer_tgt, optimizer_src,
                     device):
    model_tgt.train()
    model_src.train()
    eps_A = None
    for _ in range(args.meta_loop):
        with higher.innerloop_ctx(model_src,
                                  optimizer_src) as (fmodel_src,
                                                     foptimizer_src):
            yhat_src = fmodel_src(x_src)
            loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')
            print(f"loss_src without weight={loss_src.mean().item():.2f}")
            if eps_A is None:
                eps_A = torch.zeros(loss_src.size(),
                                    requires_grad=True).to(device)
            else:
                eps_A.requires_grad = True
            loss_src = torch.mean(eps_A * loss_src)
            print(f'loss_src weight={loss_src.item():.2f}')
            foptimizer_src.step(loss_src)
            with higher.innerloop_ctx(model_tgt,
                                      optimizer_tgt) as (fmodel_tgt,
                                                         foptimizer_tgt):
                yhat_tgt = fmodel_tgt(x_tgt)

                loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')
                yhat_src = fmodel_tgt(x_src)
                loss_src_2 = F.cross_entropy(yhat_src, y_src, reduction='none')
                loss_src_2 = loss_src_2 * args.gamma
                final_loss = torch.cat((loss_tgt, loss_src_2), dim=0)
                final_loss = torch.mean(final_loss)

                norm_sum = 0
                for sw, tw in zip(fmodel_src.parameters(),
                                  fmodel_tgt.parameters()):
                    w_diff = tw - sw
                    w_diff_norm = torch.norm(w_diff)
                    norm_sum = norm_sum + w_diff_norm**2
                norm_sum = norm_sum * args.lam
                final_loss = final_loss + norm_sum
                foptimizer_tgt.step(final_loss)

                yhat_tgt_val = fmodel_tgt(x_tgt_val)
                loss_tgt_val = F.cross_entropy(yhat_tgt_val, y_tgt_val)
                eps_A_grads = torch.autograd.grad(loss_tgt_val,
                                                  eps_A)[0].detach()
        A_tilde = torch.clamp(-eps_A_grads, min=0)
        l1_norm = torch.sum(A_tilde)
        if l1_norm != 0:
            A_tilde_min = A_tilde.min()
            A_tilde_max = A_tilde.max()
            A = (A_tilde - A_tilde_min) / (A_tilde_max - A_tilde_min)
        else:
            A = A_tilde
        print(f'A: {A}')
        eps_A = A
    return eps_A


def meta_train_ours5(args, model_tgt, model_src, x_tgt, y_tgt, x_src, y_src,
                     x_tgt_val, y_tgt_val, optimizer_tgt, optimizer_src,
                     device):
    model_tgt.train()
    model_src.train()
    eps_A = None
    eps_B = None
    for _ in range(args.meta_loop):
        with higher.innerloop_ctx(model_src,
                                  optimizer_src) as (fmodel_src,
                                                     foptimizer_src):
            yhat_src = fmodel_src(x_src)
            loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')
            print(f"loss_src without weight={loss_src.mean().item():.2f}")
            if eps_A is None:
                eps_A = torch.zeros(loss_src.size(),
                                    requires_grad=True).to(device)
            else:
                eps_A.requires_grad = True
            loss_src = torch.mean(eps_A * loss_src)
            print(f'loss_src weight={loss_src.item():.2f}')
            foptimizer_src.step(loss_src)
            with higher.innerloop_ctx(model_tgt,
                                      optimizer_tgt) as (fmodel_tgt,
                                                         foptimizer_tgt):
                yhat_tgt = fmodel_tgt(x_tgt)

                loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')
                yhat_src2 = fmodel_tgt(x_src)
                loss_src_2 = F.cross_entropy(yhat_src2,
                                             y_src,
                                             reduction='none')
                if eps_B is None:
                    eps_B = torch.zeros(loss_src_2.size(),
                                        requires_grad=True).to(device)
                else:
                    eps_B.requires_grad = True
                loss_src_2 = eps_B * loss_src_2 * args.gamma
                final_loss = torch.cat((loss_tgt, loss_src_2), dim=0)
                final_loss = torch.mean(final_loss)

                norm_sum = 0
                for sw, tw in zip(fmodel_src.parameters(),
                                  fmodel_tgt.parameters()):
                    w_diff = tw - sw
                    w_diff_norm = torch.norm(w_diff)
                    norm_sum = norm_sum + w_diff_norm**2
                norm_sum = norm_sum * args.lam
                final_loss = final_loss + norm_sum
                foptimizer_tgt.step(final_loss)

                yhat_tgt_val = fmodel_tgt(x_tgt_val)
                loss_tgt_val = F.cross_entropy(yhat_tgt_val, y_tgt_val)
                eps_A_grads, eps_B_grads = torch.autograd.grad(
                    loss_tgt_val, [eps_A, eps_B])
                eps_A_grads, eps_B_grads = eps_A_grads.detach(
                ), eps_B_grads.detach()

        A_tilde = torch.clamp(-eps_A_grads, min=0)
        l1_norm = torch.sum(A_tilde)
        if l1_norm != 0:
            A_tilde_min = A_tilde.min()
            A_tilde_max = A_tilde.max()
            A = (A_tilde - A_tilde_min) / (A_tilde_max - A_tilde_min)
        else:
            A = A_tilde
        print(f'A: {A}')
        eps_A = A
        B_tilde = torch.clamp(-eps_B_grads, min=0)
        l1_norm = torch.sum(B_tilde)
        if l1_norm != 0:
            B_tilde_min = B_tilde.min()
            B_tilde_max = B_tilde.max()
            B = (B_tilde - B_tilde_min) / (B_tilde_max - B_tilde_min)
        else:
            B = B_tilde
        print(f"B: {B}")
        eps_B = B
    return (eps_A, eps_B)
