"""
trainer.py

Train 3dgan models
"""
import pprint
import torch
from torch import optim
from torch import nn
from utils import *
import os
from pathlib import Path

import trimesh

from model import net_G, net_D

# added
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import params
from tqdm import tqdm

import wandb


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info["train_loss_G/" + key] = value

    for key, value in loss_D.items():
        scalar_info["train_loss_D/" + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info["val_loss_G/" + key] = value

    for key, value in loss_D.items():
        scalar_info["val_loss_D/" + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def trainer(args):

    config_keys = [
        "epochs",
        "batch_size",
        "soft_label",
        "adv_weight",
        "d_thresh",
        "z_dim",
        "z_dis",
        "model_save_step",
        "g_lr",
        "d_lr",
        "beta",
        "cube_len",
        "leak_value",
        "bias",
    ]

    config = {**args.__dict__, **{k: getattr(params, k) for k in config_keys}}
    pprint.pprint(config)

    wandb.init(
        entity="bugan", project="simple-pytorch-3dgan", config=config,
    )

    # added for output dir
    save_file_path = params.output_dir + "/" + args.model_name
    print(save_file_path)  # ../outputs/dcgan
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # for using tensorboard
    if args.logs:
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        writer = SummaryWriter(
            params.output_dir
            + "/"
            + args.model_name
            + "/logs_"
            + model_uid
            + "_"
            + args.logs
            + "/"
        )

    # datset define
    # dsets_path = args.input_dir + args.data_dir + "train/"
    dsets_path = args.data_dir
    # if params.cube_len == 64:
    #     dsets_path = params.data_dir + params.model_dir + "30/train64/"

    print(dsets_path)  # ../volumetric_data/chair/30/train/

    train_dsets = ShapeNetDataset(dsets_path, args, "train")
    # val_dsets = ShapeNetDataset(dsets_path, args, "val")

    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=params.batch_size, shuffle=True, num_workers=16
    )
    # val_dset_loaders = torch.utils.data.DataLoader(val_dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}
    # print (dset_len["train"])

    # model define
    D = net_D(args)
    # summary(net_D, input_size=(32, 32, 32))

    G = net_G(args)
    # print(G)
    # print(D)

    wandb.watch(G)
    wandb.watch(D)

    # summary(net_G, input_size=(params.z_dim,))

    # print total number of parameters in a model
    # x = sum(p.numel() for p in G.parameters() if p.requires_grad)
    # print (x)
    # x = sum(p.numel() for p in D.parameters() if p.requires_grad)
    # print (x)

    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    # D_solver = optim.SGD(D.parameters(), lr=params.d_lr * 100, momentum=0.9)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    D.to(params.device)
    G.to(params.device)

    # criterion_D = nn.BCELoss()
    criterion_D = nn.MSELoss()
    criterion_G = nn.L1Loss()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):

        start = time.time()

        for phase in ["train"]:
            if phase == "train":
                # if args.lrsh:
                #     D_scheduler.step()
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                # if phase == 'val':
                #     itr_val += 1

                if phase == "train":
                    itr_train += 1

                X = X.to(params.device)
                # print (X)
                # print (X.size())

                batch = X.size()[0]
                # print (batch)

                Z = generateZ(args, batch)
                # print (Z.size())

                # ============= Train the discriminator =============#
                d_real = D(X)

                fake = G(Z)

                if i == 0:
                    image_saved_path = Path(params.images_dir) / args.model_name
                    image_saved_path.mkdir(parents=True, exist_ok=True)

                    samples = fake.cpu().data[:5].squeeze().numpy()

                    fnames = []
                    for i, samp in enumerate(samples):
                        # print(i, samp)
                        mesh = trimesh.voxel.VoxelGrid(
                            trimesh.voxel.encoding.DenseEncoding(samp >= 0.5)
                        ).marching_cubes
                        fname = Path(image_saved_path) / f"{epoch:04}_{i}.obj"
                        mesh.export(fname)
                        fnames.append(fname)

                    wandb.log(
                        {
                            "generated_tree_samples": [
                                wandb.Object3D(open(fname)) for fname in fnames
                            ],
                            "epoch": epoch,
                        },
                        step=itr_train,
                    )

                d_fake = D(fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
                # print (d_fake.size(), fake_labels.size())

                if params.soft_label:
                    real_labels = (
                        torch.Tensor(batch).uniform_(0.7, 1.2).to(params.device)
                    )
                    fake_labels = torch.Tensor(batch).uniform_(0, 0.3).to(params.device)

                # print (d_real.size(), real_labels.size())
                d_real_loss = criterion_D(d_real, real_labels)

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                # no deleted
                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

                if d_total_acu < params.d_thresh:
                    D.zero_grad()
                    d_loss.backward()
                    D_solver.step()

                # =============== Train the generator ===============#

                Z = generateZ(args, batch)

                # print (X)
                fake = G(Z)  # generated fake: 0-1, X: 0/1
                d_fake = D(fake)

                adv_g_loss = criterion_D(d_fake, real_labels)
                # print (fake.size(), X.size())

                # recon_g_loss = criterion_D(fake, X)
                recon_g_loss = criterion_G(fake, X)
                # g_loss = recon_g_loss + params.adv_weight * adv_g_loss
                g_loss = adv_g_loss

                if args.local_test:
                    # print('Iteration-{} , D(x) : {:.4} , G(x) : {:.4} , D(G(x)) : {:.4}'.format(itr_train, d_loss.item(), recon_g_loss.item(), adv_g_loss.item()))
                    print(
                        "Iteration-{} , D(x) : {:.4}, D(G(x)) : {:.4}".format(
                            itr_train, d_loss.item(), adv_g_loss.item()
                        )
                    )

                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                G_solver.step()

                # =============== logging each 10 iterations ===============#

                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)

                if args.logs:
                    loss_G = {
                        "adv_loss_G": adv_g_loss,
                        "recon_loss_G": recon_g_loss,
                    }

                    loss_D = {
                        "adv_real_loss_D": d_real_loss,
                        "adv_fake_loss_D": d_fake_loss,
                        "d_real_acu": d_real_acu.mean(),
                        "d_fake_acu": d_fake_acu.mean(),
                        "d_total_acu": d_total_acu,
                    }

                    # if itr_val % 10 == 0 and phase == 'val':
                    #     save_val_log(writer, loss_D, loss_G, itr_val)

                    if itr_train % 10 == 0 and phase == "train":
                        save_train_log(writer, loss_D, loss_G, itr_train)

                        wandb.log(
                            {"G": loss_G, "D": loss_D, "epoch": epoch}, step=itr_train
                        )

            # =============== each epoch save model or save image ===============#
            epoch_loss_G = running_loss_G / dset_len[phase]
            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]

            end = time.time()
            epoch_time = end - start

            print(
                "Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}".format(
                    epoch, phase, epoch_loss_D, epoch_loss_adv_G
                )
            )
            print("Elapsed Time: {:.4} min".format(epoch_time / 60.0))

            if (epoch + 1) % params.model_save_step == 0:

                print("model_saved, images_saved...")
                torch.save(
                    G.state_dict(),
                    params.output_dir + "/" + args.model_name + "/" + "G" + ".pth",
                )
                torch.save(
                    D.state_dict(),
                    params.output_dir + "/" + args.model_name + "/" + "D" + ".pth",
                )

                # SavePloat_Voxels(samples, image_saved_path, epoch)
