"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 12, 2025
 * Purpose: Provides utility functions for the 3D-VAE-GAN model, including
            data loading, tensor manipulation, checkpoint saving/loading,
            and 3D voxel visualization tools.
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
import json
import torch
import os
import pickle
from PIL import Image
from torchvision import transforms
from src.preprocessing import binvox_rw

class CO3DDataset(data.Dataset):
    def __init__(self, root, args, multiview=False, use_pose=False, apply_mask=False):
        """
        CO3D dataset class that supports single/multi-view and with/without pose options.

        Args:
            root: Root directory containing category folders
            args: Arguments containing image_size, etc.
            multiview: Whether to use multiple views (True) or single view (False)
            use_pose: Whether to include camera pose information
        """
        self.root = root
        self.args = args
        self.multiview = multiview
        self.use_pose = use_pose
        self.img_size = args.image_size
        self.num_views = args.num_views if multiview else 1
        self.apply_mask = apply_mask  # Whether to apply mask to images
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        # Find all sequences (each sequence is a 3D object)
        self.category_dir = os.path.join(root, args.obj)
        self.sequences = []

        # Get all dataset instance directories
        for item in os.listdir(self.category_dir):
            if item.endswith('_dataset_instance'):
                seq_name = item.replace('_dataset_instance', '')
                processed_dir = os.path.join(self.category_dir, f"{seq_name}_processed")
                instance_dir = os.path.join(self.category_dir, item)

                # Check if both processed and dataset instance directories exist
                if os.path.isdir(processed_dir) and os.path.isdir(instance_dir):
                    # Check if binvox file exists
                    binvox_path = os.path.join(instance_dir, f"{seq_name}.binvox")
                    if os.path.exists(binvox_path):
                        # Check if metadata exists
                        meta_path = os.path.join(processed_dir, "sequence_meta.json")
                        if os.path.exists(meta_path):
                            self.sequences.append({
                                "name": seq_name,
                                "processed_dir": processed_dir,
                                "instance_dir": instance_dir,
                                "binvox_path": binvox_path,
                                "meta_path": meta_path
                            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]

        # Load the voxel data
        with open(sequence["binvox_path"], 'rb') as f:
            volume = self.load_binvox(f)

        # Load metadata
        with open(sequence["meta_path"], 'r') as f:
            metadata = json.load(f)

        # Get frames information
        frames = metadata["frames"]

        # Select frames based on viewing mode
        if self.multiview:
            # Select multiple frames for multi-view
            if len(frames) > self.num_views:
                # Evenly sample frames
                indices = np.linspace(0, len(frames) - 1, self.num_views, dtype=int)
                selected_frames = [frames[i] for i in indices]
            else:
                # Use all available frames
                selected_frames = frames[:self.num_views]
        else:
            # For single-view, just use the first frame
            selected_frames = [frames[0]]

        # Process frames
        images = []
        poses = [] if self.use_pose else None

        for frame in selected_frames:
            # Load and process image
            img_path = os.path.join(sequence["instance_dir"], frame["image"])
            img = Image.open(img_path).convert('RGB')

            # Load and apply mask if needed
            if self.apply_mask:
                mask_path = os.path.join(sequence["instance_dir"], frame["mask"])
                mask = Image.open(mask_path).convert('L')
                mask_np = np.array(mask) > 128
                img_np = np.array(img)

                # Apply mask
                for c in range(3):
                    img_np[:, :, c] = np.where(mask_np, img_np[:, :, c], 0)

                img = Image.fromarray(img_np)

            # Transform and add to list
            img_tensor = self.transform(img)
            images.append(img_tensor)

            # Load pose if needed
            if self.use_pose:
                pose_path = os.path.join(sequence["processed_dir"], frame["pose"])
                pose = self.load_pose(pose_path)
                poses.append(pose)

        # Return appropriate tuple based on configuration
        volume_tensor = torch.FloatTensor(volume)

        if self.multiview:
            if self.use_pose:
                return images, poses, volume_tensor
            else:
                return images, volume_tensor
        else:
            # Single view - return first image/pose directly rather than in a list
            if self.use_pose:
                return images[0], poses[0], volume_tensor
            else:
                return images[0], volume_tensor

    def load_pose(self, pose_path):
        """Load camera pose from JSON file"""
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        R = torch.FloatTensor(pose_data["R"])
        T = torch.FloatTensor(pose_data["T"])
        return {'R': R, 'T': T}

    def load_binvox(self, f):
        """Load binvox file and return as numpy array"""
        model = binvox_rw.read_as_3d_array(f)
        return np.asarray(model.data, dtype=np.float32)


########################## HELPER METHODS ###############################

def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig('test')
    plt.show()


def SavePloat_Voxels(voxels, path, iteration):
    print(f"Voxels shape: {voxels.shape}")  # Add this line to debug\
    # Ensure we have the right format
    if len(voxels.shape) == 5:  # (batch, channels, depth, height, width)
        # Remove channel dimension if it's 1
        if voxels.shape[1] == 1:
            voxels = voxels.squeeze(1)  # Now shape (batch, depth, height, width)

    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    #plt.show()
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def generateZ(args):
    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))


    except Exception as e:
        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None,E_solver = None):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)