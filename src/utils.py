"""
utils.py

Some utility functions

"""
import os
import functools
import scipy.ndimage as nd
import scipy.io as io
import matplotlib
from . import params
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
import tqdm
import wandb

from joblib import delayed, Memory, Parallel

memory = Memory("/tmp/.joblib.cache", verbose=0)
np.random.seed(123)


def mesh2arrayCentered(mesh, array_length=32, voxel_size=1):
    # given array length 64, voxel size 2, then output array size is [128,128,128]
    resolution = np.ceil(
        np.array([array_length, array_length, array_length]) / voxel_size
    ).astype(int)
    vox_array = np.zeros(
        resolution, dtype=bool
    )  # tanh: voxel representation [-1,1], sigmoid: [0,1]
    # scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(
        trimesh.transformations.scale_matrix((array_length - 1.5) / max_length)
    )  # now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  # max voxel array length = array_length / voxel_size

    # find indices in the v.matrix to center it in vox_array
    indices = ((resolution - v.matrix.shape) / 2).astype(int)
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


@memory.cache()
def load_mesh(path):
    return trimesh.load(path, force="mesh")


def random_rotate(mesh):
    mesh = mesh.copy()
    angle_rad = np.random.rand() * 2 * np.pi
    direction = trimesh.unitize(np.random.rand(3))
    rot = trimesh.transformations.rotation_matrix(angle_rad, direction, [0, 0, 0])
    return mesh.apply_transform(rot)


def voxel_from_obj_file(path, rotate=True, res=32):
    mesh = load_mesh(path)

    if rotate:
        # 2 random rotations
        mesh_rot = random_rotate(mesh)
        mesh_rot = random_rotate(mesh_rot)
        try:
            voxels = mesh2arrayCentered(mesh_rot, array_length=res)
        except ValueError:
            # use the original mesh
            voxels = mesh2arrayCentered(mesh, array_length=res)
    else:
        voxels = mesh2arrayCentered(mesh, array_length=res)

    volume = np.asarray(voxels, dtype=np.float32)
    return torch.FloatTensor(volume)


@memory.cache()
def load_deterministic(path, res=32):
    return voxel_from_obj_file(path, rotate=False, res=res)


def check(path):
    try:
        load_deterministic(path)
        return True
    except Exception:
        return False


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


class AugmentDataset(data.Dataset):
    def __init__(self, root, args, train_or_val="train", res=32):

        self.root = root
        paths = [
            p for p in Path(root).rglob("*.*") if p.suffix.lower() in [".obj", ".ply"]
        ]

        loadable = Parallel(verbose=2, n_jobs=-1)(
            delayed(check)(path) for path in paths
        )

        self.listdir = [p for p, is_loadable in zip(paths, loadable) if is_loadable]

        print(f"Using {len(self.listdir)} of {len(paths)} files.")

        self.args = args
        self.res = res

    def __getitem__(self, index):
        fname = self.listdir[index]
        return voxel_from_obj_file(fname, rotate=True, res=self.res)

    def __len__(self):
        return len(self.listdir)


class ShapeNetDataset(data.Dataset):
    def __init__(self, root, args, train_or_val="train", res=32):

        self.root = root
        paths = [
            p for p in Path(root).rglob("*.*") if p.suffix.lower() in [".obj", ".ply"]
        ]

        loadable = Parallel(verbose=2, n_jobs=-1)(
            delayed(check)(path) for path in paths
        )

        loadable_paths = [path for path, can_load in zip(paths, loadable) if can_load]

        # self.listdir = [p for p, is_loadable in zip(paths, loadable) if is_loadable]

        self.voxels = Parallel(verbose=2, n_jobs=-1)(
            delayed(load_deterministic)(path, res) for path in loadable_paths
        )
        # print (self.listdir)
        # print(f"Using {len(self.listdir)} of {len(paths)} files.")

        # data_size = len(self.listdir)
        #        self.listdir = self.listdir[0:int(data_size*0.7)]

        self.args = args
        self.res = res

    def __getitem__(self, index):
        return self.voxels[index]
        # fname = self.listdir[index]
        # return voxel_from_obj_file(fname, rotate=args.rotate)

    def __len__(self):
        # return len(self.listdir)
        return len(self.voxels)


def generateZ(args, batch):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z


def save_model(run, model_name, G, D):
    path_G = params.output_dir + "/" + model_name + "/G.pth"
    path_D = params.output_dir + "/" + model_name + "/D.pth"
    torch.save(
        G.state_dict(),
        path_G,
    )
    torch.save(
        D.state_dict(),
        path_D,
    )

    # save also in wandb
    path_G = str(Path(run.dir).absolute() / "G.pth")
    path_D = str(Path(run.dir).absolute() / "D.pth")
    torch.save(
        G.state_dict(),
        path_G,
    )
    torch.save(
        D.state_dict(),
        path_D,
    )
    wandb.save(path_G)
    wandb.save(path_D)


def load_model(run, G, D):
    file_G = wandb.restore("G.pth").name
    file_D = wandb.restore("D.pth").name

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(file_G, map_location={"cuda:0": "cpu"}))
        D.load_state_dict(torch.load(file_D, map_location={"cuda:0": "cpu"}))
    else:
        G.load_state_dict(torch.load(file_G))
        D.load_state_dict(torch.load(file_D))

    return G, D
