"""
utils.py

Some utility functions

"""
import os
import functools
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import params
from pathlib import Path

if params.device.type != "cpu":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle

import trimesh
import numpy as np

# from joblib import Memory
# memory = Memory("../.joblib.cache", verbose=0)


def mesh2arrayCentered(mesh, voxel_size=1, array_length=32):
    # given array length 64, voxel size 2, then output array size is [128,128,128]
    array_size = np.ceil(
        np.array([array_length, array_length, array_length]) / voxel_size
    ).astype(int)
    vox_array = np.zeros(
        array_size, dtype=bool
    )  # tanh: voxel representation [-1,1], sigmoid: [0,1]
    # scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(
        trimesh.transformations.scale_matrix((array_length - 1) / max_length)
    )  # now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  # max voxel array length = array_length / voxel_size

    # find indices in the v.matrix to center it in vox_array
    indices = ((array_size - v.matrix.shape) / 2).astype(int)
    vox_array[
        indices[0] : indices[0] + v.matrix.shape[0],
        indices[1] : indices[1] + v.matrix.shape[1],
        indices[2] : indices[2] + v.matrix.shape[2],
    ] = v.matrix

    return vox_array


def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)["instance"]  # 30x30x30
        voxels = np.pad(voxels, (1, 1), "constant", constant_values=(0, 0))

    else:
        # voxels = np.load(path)
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)["instance"]  # 30x30x30
        voxels = np.pad(voxels, (1, 1), "constant", constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode="constant", order=0)
        # print ('here')
    # print (voxels.shape)
    return voxels


# @memory.cache()
def voxel_from_obj_file(path):
    mesh = trimesh.load(path, force="mesh")
    voxels = mesh2arrayCentered(mesh)
    volume = np.asarray(voxels, dtype=np.float32)
    return torch.FloatTensor(volume)


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection="3d")
        ax.scatter(x, y, z, zdir="z", c="red")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + "/{}.png".format(str(iteration).zfill(3)), bbox_inches="tight")
    plt.close()


class ShapeNetDataset(data.Dataset):
    def __init__(self, root, args, train_or_val="train"):

        self.root = root
        self.listdir = list(Path(root).rglob("*.obj"))
        # print (self.listdir)
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
        #        self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0 : int(data_size)]

        print("data_size =", len(self.listdir))  # train: 10668-1000=9668
        self.args = args

    def __getitem__(self, index):
        fname = self.listdir[index]
        return voxel_from_obj_file(fname)

    def __len__(self):
        return len(self.listdir)


def generateZ(args, batch):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z
