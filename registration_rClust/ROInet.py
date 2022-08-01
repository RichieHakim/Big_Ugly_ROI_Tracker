import sys
from pathlib import Path
import json
import os
import hashlib
import PIL

import numpy as np
import gdown
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal


class ROInet_embedder:
    def __init__(
        self,
        device='cpu',
        dir_networkFiles=None,
        download_from_gDrive='check_local_first',
        gDriveID='1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8',
        hash_dict_networkFiles={
            'params': ('params.json', '877e17df8fa511a03bc99cd507a54403'),
            'model': ('model.py', '741b79903507b11769e3f7aa4cdd4dbe'),
            'state_dict': ('ConvNext_tiny__1_0_unfrozen__simCLR.pth', 'a5fae4c9ea95f2c78b4690222b2928a5'),
        },
        verbose=True,
    ):
        """
        Initialize the class.

        Args:
            device (str):
                The device to use for the model and data.
            dir_networkFiles (str):
                The directory to find or download the network files into
            download_from_gDrive (str):
                Approach to downloading the network files.
                'check_local_first':
                    Check if the network files are already in 
                     dir_networkFiles. If so, use them.
                'force':
                    Download the network files from Google Drive.
            hash_dict_networkFiles (dict):
                A dictionary of the hash values of the network files.
                Each item is {key: (filename, hash_value)}
                The (filename, hash_value) pairs can be made using:
                 paths_networkFiles = [(Path(dir_networkFiles).resolve() / name).as_posix() for name in get_dir_contents(dir_networkFiles)[1]]
                 {Path(path).name: hash_file(path) for path in paths_networkFiles}
            verbose (bool):
                Whether to print out extra information.
        """
        self.device = device
        self.verbose = verbose


        self.dir_networkFiles = dir_networkFiles
        self.gDriveID = gDriveID

        ## Find or download network files
        if download_from_gDrive == 'force':
            self._download_network_files()
            if hash_dict_networkFiles is None:
                print('Skipping hash check because hash_dict_networkFiles is None')
            else:
                results_all, results, paths_matching = compare_file_hashes(  
                    hash_dict_true=hash_dict_networkFiles,
                    dir_files_test=dir_networkFiles,
                    verbose=True,
                )
                if results_all == False:
                    print(f'WARNING: Hash comparison failed. Could not match downloaded files to hash_dict_networkFiles.')

        if download_from_gDrive == 'check_local_first':
            assert hash_dict_networkFiles is not None, "if using download_from_gDrive='check_local_first' hash_dict_networkFiles cannot be None"
            results_all, results, paths_matching = compare_file_hashes(  
                hash_dict_true=hash_dict_networkFiles,
                dir_files_test=dir_networkFiles,
                verbose=True,
            )
            print(f'Successful hash comparison. Found matching files: {paths_matching}') if results_all and self.verbose else None
            if results_all == False:
                print(f'Hash comparison failed. Downloading from Google Drive.') if self.verbose else None
                self._download_network_files()
                results_all, results, paths_matching = compare_file_hashes(  
                    hash_dict_true=hash_dict_networkFiles,
                    dir_files_test=dir_networkFiles,
                    verbose=True,
                )
                if results_all:
                    print(f'Successful hash comparison. Found matching files: {paths_matching}')  if self.verbose else None
                else:
                    raise Exception(f'Downloaded network files do not match expected hashes. Results: {results}')

        ## Import network files
        sys.path.append(dir_networkFiles)
        import model
        print(f"Imported model from {dir_networkFiles}/model.py") if self.verbose else None

        with open(paths_matching['params']) as f:
            self.params_model = json.load(f)
            print(f"Loaded params_model from {paths_matching['params']}") if self.verbose else None
            self.net = model.make_model(self.params_model)
            print(f"Generated network using params_model") if self.verbose else None
            
        ## Prep network and load state_dict
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        self.net.load_state_dict(torch.load(paths_matching['state_dict']))
        print(f'Loaded state_dict into network from {paths_matching["state_dict"]}') if self.verbose else None

        self.net = self.net.to(self.device)
        print(f'Loaded network onto device {self.device}') if self.verbose else None

        ## Prepare dataloader
        
    def generate_dataloader(
        self,
        ROI_images,
        goal_size=250,
        ptile_norm=90,
        scale_norm=0.6,
        pref_plot=False,
        batchSize_dataloader=8,
        pinMemory_dataloader=True,
        numWorkers_dataloader=-1,
        persistentWorkers_dataloader=True,
        prefetchFactor_dataloader=2,
    ):
        """
        Generate a dataloader for the given ROI_images.
        """
        print('Starting: resizing ROIs') if self.verbose else None
        sf_ptiles = np.array([np.percentile(np.sum(sf>0, axis=(1,2)), ptile_norm) for sf in tqdm(ROI_images)])
        scales_forRS = (goal_size/sf_ptiles)**scale_norm
        sf_rs = [np.stack([resize_affine(img, scale=scales_forRS[ii], clamp_range=True) for img in sf], axis=0) for ii, sf in enumerate(tqdm(ROI_images))]

        ROI_images_cat = np.concatenate(ROI_images, axis=0)
        ROI_images_rs = np.concatenate(sf_rs, axis=0)
        print('Completed: resizing ROIs') if self.verbose else None

        if pref_plot:
            fig, axs = plt.subplots(2,1, figsize=(7,10))
            axs[0].plot(np.sum(ROI_images_cat > 0, axis=(1,2)))
            axs[0].plot(scipy.signal.savgol_filter(np.sum(ROI_images_cat > 0, axis=(1,2)), 501, 3))
            axs[0].set_xlabel('ROI number');
            axs[0].set_ylabel('mean npix');
            axs[0].set_title('ROI sizes raw')

            axs[1].plot(np.sum(ROI_images_rs > 0, axis=(1,2)))
            axs[1].plot(scipy.signal.savgol_filter(np.sum(ROI_images_rs > 0, axis=(1,2)), 501, 3))
            axs[1].set_xlabel('ROI number');
            axs[1].set_ylabel('mean npix');
            axs[1].set_title('ROI sizes resized')

        transforms = torch.nn.Sequential(
            ScaleDynamicRange(scaler_bounds=(0,1)),
            torchvision.transforms.Resize(
                size=(224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ), 
            TileChannels(dim=0, n_channels=3),
        )
        transforms_scripted = torch.jit.script(transforms)
        print(f'Defined image transformations: {transforms}') if self.verbose else None


        self.dataset = dataset_simCLR(
                X=torch.as_tensor(ROI_images_rs, device='cpu', dtype=torch.float32),
                y=torch.as_tensor(torch.zeros(ROI_images_rs.shape[0]), device='cpu', dtype=torch.float32),
                n_transforms=1,
                class_weights=np.array([1]),
                transform=transforms_scripted,
                DEVICE='cpu',
                dtype_X=torch.float32,
            )
        print(f'Defined dataset') if self.verbose else None
            
        self.dataloader = torch.utils.data.DataLoader( 
                self.dataset,
                batch_size=batchSize_dataloader,
                shuffle=False,
                drop_last=False,
                pin_memory=pinMemory_dataloader,
                num_workers=numWorkers_dataloader,
                persistent_workers=persistentWorkers_dataloader,
                prefetch_factor=prefetchFactor_dataloader,
        )
        print(f'Defined dataloader') if self.verbose else None

    def generate_latents(self):
        """
        Pass the data in the dataloader through the network and generate latents.

        Returns:
            latents (torch.Tensor): 
                latents for each ROI
        """
        if hasattr(self, 'dataloader') == False:
            raise Exception('dataloader not defined. Call generate_dataloader() first.')

        print(f'starting: running data through network')
        self.latents = torch.cat([self.net(data[0][0].to(self.device)).detach() for data in tqdm(self.dataloader)], dim=0).cpu()
        print(f'completed: running data through network')
        return self.latents


    def _download_network_files(self):
        if self.gDriveID is None or self.dir_networkFiles is None:
            raise ValueError('gDriveID and dir_networkFiles must be specified if download_from_gDrive is True')

        self.gDriveID = self.gDriveID
        self.dir_networkFiles = self.dir_networkFiles

        print(f'Downloading network files from Google Drive to {self.dir_networkFiles}') if self.verbose else None
        gdown.download_folder(id=self.gDriveID, output=self.dir_networkFiles, quiet=False, use_cookies=False)
        print('Downloaded network files') if self.verbose else None
    

def resize_affine(img, scale, clamp_range=False):
    """
    Wrapper for torchvision.transforms.Resize.
    Useful for resizing images to match the size of the images
     used in the training of the neural network.
    RH 2022

    Args:
        img (np.ndarray): 
            Image to resize
            shape: (H,W)
        scale (float):
            Scale factor to use for resizing
        clamp_range (bool):
            If True, will clamp the image to the range [min(img), max(img)]
            This is to prevent the interpolation from going outside of the
             range of the image.

    Returns:
        np.ndarray:
            Resized image
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
#         img=torch.as_tensor(img[None,...]),
        img=PIL.Image.fromarray(img),
        angle=0, translate=[0,0], shear=0,
        scale=scale,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    ))
    
    if clamp_range:
        clamp_high = img.max()
        clamp_low = img.min()
    
        img_rs[img_rs>clamp_high] = clamp_high
        img_rs[img_rs<clamp_low] = clamp_low
    
    return img_rs


###################################
########### FROM BNPM #############
###################################

def get_dir_contents(directory):
    '''
    Get the contents of a directory (does not
     include subdirectories).
    RH 2021

    Args:
        directory (str):
            path to directory
    
    Returns:
        folders (List):
            list of folder names
        files (List):
            list of file names
    '''
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files



def hash_file(path, type_hash='MD5', buffer_size=65536):
    """
    Gets hash of a file.
    Based on: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    RH 2022

    Args:
        path (str):
            Path to file to be hashed.
        type_hash (str):
            Type of hash to use. Can be:
                'MD5'
                'SHA1'
                'SHA256'
                'SHA512'
        buffer_size (int):
            Buffer size for reading file.
            65536 corresponds to 64KB.

    Returns:
        hash (str):
            Hash of file.
    """

    if type_hash == 'MD5':
        hasher = hashlib.md5()
    elif type_hash == 'SHA1':
        hasher = hashlib.sha1()
    elif type_hash == 'SHA256':
        hasher = hashlib.sha256()
    elif type_hash == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{type_hash} is not a valid hash type.')

    with open(path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)

    hash = hasher.hexdigest()
        
    return hash


def compare_file_hashes(
    hash_dict_true,
    dir_files_test=None,
    paths_files_test=None,
    verbose=True,
):
    """
    Compares hashes of files in a directory or list of paths
     to user provided hashes.
    RH 2022

    Args:
        hash_dict_true (dict):
            Dictionary of hashes to compare to.
            Each entry should be:
                {'key': ('filename', 'hash')}
        dir_files_test (str):
            Path to directory to compare hashes of files in.
            Unused if paths_files_test is not None.
        paths_files_test (list of str):
            List of paths to files to compare hashes of.
            Optional. dir_files_test is used if None.
        verbose (bool):
            Whether or not to print out failed comparisons.

    Returns:
        total_result (bool):
            Whether or not all hashes were matched.
        individual_results (list of bool):
            Whether or not each hash was matched.
        paths_matching (dict):
            Dictionary of paths that matched.
            Each entry is:
                {'key': 'path'}
    """
    if paths_files_test is None:
        if dir_files_test is None:
            raise ValueError('Must provide either dir_files_test or path_files_test.')
        
        ## make a dict of {filename: path} for each file in dir_files_test
        files_test = {filename: (Path(dir_files_test).resolve() / filename).as_posix() for filename in get_dir_contents(dir_files_test)[1]} 
    
    paths_matching = {}
    results_matching = {}
    for key, (filename, hash) in hash_dict_true.items():
        match = True
        if filename not in files_test:
            print(f'{filename} not found in test directory: {dir_files_test}.') if verbose else None
            match = False
        elif hash != hash_file(files_test[filename]):
            print(f'{filename} hash mismatch with {key, filename}.') if verbose else None
            match = False
        if match:
            paths_matching[key] = files_test[filename]
        results_matching[key] = match

    return all(results_matching.values()), results_matching, paths_matching



###################################
########### FROM GRC ##############
###################################

class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=0, n_channels=3):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"
        
class ScaleDynamicRange(Module):
    """
    Min-max scaling of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        
        self.epsilon = epsilon
    
    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max()+self.epsilon))
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.bounds})"


class dataset_simCLR(Dataset):
    """    
    demo:
    
    transforms = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    
    torchvision.transforms.GaussianBlur(5,
                                        sigma=(0.01, 1.)),
    
    torchvision.transforms.RandomPerspective(distortion_scale=0.6, 
                                             p=1, 
                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                             fill=0),
    torchvision.transforms.RandomAffine(
                                        degrees=(-180,180),
                                        translate=(0.4, 0.4),
                                        scale=(0.7, 1.7), 
                                        shear=(-20, 20, -20, 20), 
                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                        fill=0, 
                                        fillcolor=None, 
                                        resample=None),
    )
    scripted_transforms = torch.jit.script(transforms)

    dataset = util.dataset_simCLR(  torch.tensor(images), 
                                labels, 
                                n_transforms=2, 
                                transform=scripted_transforms,
                                DEVICE='cpu',
                                dtype_X=torch.float32,
                                dtype_y=torch.int64 )
    
    dataloader = torch.utils.data.DataLoader(   dataset,
                                            batch_size=64,
        #                                     sampler=sampler,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=False,
                                            num_workers=0,
                                            )
    """
    def __init__(   self, 
                    X, 
                    y, 
                    n_transforms=2,
                    class_weights=None,
                    transform=None,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64,
                    temp_uncertainty=1,
                    expand_dim=True
                    ):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021 / JZ 2021

        Args:
            X (torch.Tensor / np.array / list of float32):
                Images.
                Shape: (n_samples, height, width)
                Currently expects no channel dimension. If/when
                 it exists, then shape should be
                (n_samples, n_channels, height, width)
            y (torch.Tensor / np.array / list of ints):
                Labels.
                Shape: (n_samples)
            n_transforms (int):
                Number of transformations to apply to each image.
                Should be >= 1.
            transform (callable, optional):
                Optional transform to be applied on a sample.
                See torchvision.transforms for more information.
                Can use torch.nn.Sequential( a bunch of transforms )
                 or other methods from torchvision.transforms. Try
                 to use torch.jit.script(transform) if possible.
                If not None:
                 Transform(s) are applied to each image and the 
                 output shape of X_sample_transformed for 
                 __getitem__ will be
                 (n_samples, n_transforms, n_channels, height, width)
                If None:
                 No transform is applied and output shape
                 of X_sample_trasformed for __getitem__ will be 
                 (n_samples, n_channels, height, width)
                 (which is missing the n_transforms dimension).
            DEVICE (str):
                Device on which the data will be stored and
                 transformed. Best to leave this as 'cpu' and do
                 .to(DEVICE) on the data for the training loop.
            dtype_X (torch.dtype):
                Data type of X.
            dtype_y (torch.dtype):
                Data type of y.
        
        Returns:
            torch.utils.data.Dataset:
                torch.utils.data.Dataset object.
        """

        self.expand_dim = expand_dim
        
        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE) # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.X = self.X[:,None,...] if expand_dim else self.X
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.
        
        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        self.n_transforms = n_transforms

        self.temp_uncertainty = temp_uncertainty

        self.headmodel = None

        self.net_model = None
        self.classification_model = None
        
        
        # self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=DEVICE)

        # self.classModelParams_coef_ = mp.Array(np.ctypeslib.as_array(mp.Array(ctypes.c_float, feature)))

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')
    
    def tile_channels(X_in, dim=-3):
        """
        Expand dimension dim in X_in and tile to be 3 channels.

        JZ 2021 / RH 2021

        Args:
            X_in (torch.Tensor or np.ndarray):
                Input image. 
                Shape: [n_channels==1, height, width]

        Returns:
            X_out (torch.Tensor or np.ndarray):
                Output image.
                Shape: [n_channels==3, height, width]
        """
        dims = [1]*len(X_in.shape)
        dims[dim] = 3
        return torch.tile(X_in, dims)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves and transforms a sample.
        RH 2021 / JZ 2021

        Args:
            idx (int):
                Index / indices of the sample to retrieve.
            
        Returns:
            X_sample_transformed (torch.Tensor):
                Transformed sample(s).
                Shape: 
                    If transform is None:
                        X_sample_transformed[batch_size, n_channels, height, width]
                    If transform is not None:
                        X_sample_transformed[n_transforms][batch_size, n_channels, height, width]
            y_sample (int):
                Label(s) of the sample(s).
            idx_sample (int):
                Index of the sample(s).
        """

        y_sample = self.y[idx]
        idx_sample = self.idx[idx]
        
        if self.classification_model is not None:
            # features = self.net_model(tile_channels(self.X[idx][:,None,...], dim=1))
            # proba = self.classification_model.predict_proba(features.cpu().detach())[0]
            proba = self.classification_model.predict_proba(self.tile_channels(self.X[idx_sample][:,None,...], dim=-3))[0]
            
            # sample_weight = loss_uncertainty(torch.as_tensor(proba, dtype=torch.float32), temperature=self.temp_uncertainty)
            sample_weight = 1
        else:
            sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):

                # X_sample_transformed.append(tile_channels(self.transform(self.X[idx_sample]), dim=0))
                X_transformed = self.transform(self.X[idx_sample])
                X_sample_transformed.append(X_transformed)
        else:
            X_sample_transformed = self.tile_channels(self.X[idx_sample], dim=-3)
        
        return X_sample_transformed, y_sample, idx_sample, sample_weight


