from pathlib import Path
import scipy.io
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold
import sklearn.cluster
import seaborn as sns
import pandas as pd
import sparse
import cv2

import torch
import spconv.pytorch as spconv


import gc
from tqdm.notebook import tqdm
import copy
import time
import random

from .bnpm_helpers import *

def apply_CLAHE(images, clipLimit=40):
    clahe = cv2.createCLAHE(clipLimit = clipLimit)
    ims_clahe = [(255*im/im.max()).astype(np.uint8) for im in images]
    return ims_clahe


def display_toggle_image_stack(images, clim=None):
    from ipywidgets import interact, widgets

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imshow_FOV = ax.imshow(
        images[0],
#         vmax=clim[1]
    )

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        imshow_FOV.set_data(images[i_frame])
        imshow_FOV.set_clim(clim)


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def display_toggle_2channel_image_stack(images, clim=None):
    from ipywidgets import interact, widgets

    fig, axs = plt.subplots(1,2 , figsize=(14,8))
    ax_1 = axs[0].imshow(images[0][...,0], clim=clim)
    ax_2 = axs[1].imshow(images[0][...,1], clim=clim)

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        ax_1.set_data(images[i_frame][...,0])
        ax_2.set_data(images[i_frame][...,1])


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def import_and_convert_to_CellReg_spatialFootprints(
    paths_statFiles, 
    frame_height=512, 
    frame_width=1024,
    dtype=np.uint8,
    ):
    """
    Imports and converts multiple stat files to spatial footprints
     suitable for CellReg.
    Output will be a list of arrays of shape (n_roi, height, width).
    RH 2022
    """

    isInt = np.issubdtype(dtype, np.integer)

    stats = [np.load(path, allow_pickle=True) for path in paths_statFiles]
    num_rois = [stat.size for stat in stats]
    sf_all_list = [np.zeros((n_roi, frame_height, frame_width), dtype) for n_roi in num_rois]
    for ii, stat in enumerate(stats):
        for jj, roi in enumerate(stat):
            lam = np.array(roi['lam'])
            if isInt:
                lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
            else:
                lam = lam / lam.sum()
            sf_all_list[ii][jj, roi['ypix'], roi['xpix']] = lam
    return sf_all_list


def register_ROIs(templateFOV, FOVs, ROIs, return_sparse=False, normalize=True):
    dims = templateFOV.shape
    x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))

    template_norm = np.uint8(templateFOV * (templateFOV > 0) * (1/templateFOV.max()) * 255)
    FOVs_norm    = [np.uint8(FOVs[ii] * (FOVs[ii] > 0) * (1/FOVs[ii].max()) * 255) for ii in range(len(FOVs))]

    def safe_ROI_remap(img_ROI, x_remap, y_remap):
        img_ROI_remap = cv2.remap(img_ROI.astype(np.float32), x_remap, y_remap, cv2.INTER_LINEAR)
        if img_ROI_remap.sum() == 0:
            img_ROI_remap = img_ROI
        return img_ROI_remap

    
    ROIs_aligned, FOVs_aligned, flows = [], [], []
    for ii in range(len(FOVs)):
#         flow = cv2.calcOpticalFlowFarneback(template_norm, FOVs_norm[ii], None,
#                                             0.5, 3, 128, 3, 7, 1.5, 0)
#         flow = cv2.calcOpticalFlowFarneback(
#             prev=template_norm,
#             next=FOVs_norm[ii], 
#             flow=None, 
#             pyr_scale=0.3, 
#             levels=3,
#             winsize=128, 
#             iterations=7,
#             poly_n=7, 
#             poly_sigma=1.5,
#             flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
#         )
    
        flow = cv2.optflow.createOptFlow_DeepFlow().calc(
            template_norm,
            FOVs_norm[ii],
            None
        )
            
        x_remap = (flow[:, :, 0] + x_grid).astype(np.float32)
        y_remap = (flow[:, :, 1] + y_grid).astype(np.float32)

        ROI_aligned = np.stack([safe_ROI_remap(img.astype(np.float32), x_remap, y_remap) for img in ROIs[ii]], axis=0)
#         ROI_aligned = np.stack([img.astype(np.float32) for img in ROIs[ii]], axis=0)
        FOV_aligned = cv2.remap(FOVs_norm[ii], x_remap, y_remap, cv2.INTER_NEAREST)

        if normalize:
            ROI_aligned = ROI_aligned / np.sum(ROI_aligned, axis=(1,2), keepdims=True)
        
        if return_sparse:
            ROIs_aligned.append(scipy.sparse.csc_matrix(ROI_aligned.reshape(ROI_aligned.shape[0], -1)))
            FOVs_aligned.append(FOV_aligned)
            flows.append(flow)
        else:
            ROIs_aligned.append(ROI_aligned)
            FOVs_aligned.append(FOV_aligned)
            flows.append(flow)
    return ROIs_aligned, FOVs_aligned, flows


########################
## Sparse convolution ##
########################

def pydata_sparse_to_spconv(sp_array, device='cpu'):
    coo = sparse.COO(sp_array)
    idx_raw = torch.as_tensor(coo.coords.T, dtype=torch.int32, device=device).contiguous()
    spconv_array = spconv.SparseConvTensor(
        features=torch.as_tensor(coo.reshape((-1)).T.data, dtype=torch.float32, device=device)[:,None].contiguous(),
        indices=idx_raw,
        spatial_shape=coo.shape[1:], 
        batch_size=coo.shape[0]
    )
    return spconv_array

def sparse_convert_spconv_to_scipy(sp_arr):
    coo = sparse.COO(
        coords=sp_arr.indices.T.to('cpu'),
        data=sp_arr.features.squeeze().to('cpu'),
        shape=[sp_arr.batch_size] + sp_arr.spatial_shape
    )
    return coo.reshape((coo.shape[0], -1)).to_scipy_sparse().tocsr()
    

def sparse_conv2D(images, kernel, image_shape, device='cpu'):
    """
    images: scipy.sparse.csr_matrix. Flattened images.
        an individual image should be reconstructable by:
         images[ii].toarray().reshape(image_shape[0], image_shape[1])
    kernel: 2d array
    """
    
    ## prepare images
    images_spconv = pydata_sparse_to_spconv(
        sparse.COO(images).reshape((images.shape[0], image_shape[0], image_shape[1])),
        device=device
    )
    
    ## prepare kernel
    kernel_prep = torch.as_tensor(
        kernel[:,:,None,None], 
        dtype=torch.float32,
        device=device
    ).contiguous()
    
    ## prepare convolution
    conv = spconv.SparseConv2d(
        in_channels=1, 
        out_channels=1,
        kernel_size=kernel.shape, 
        stride=1, 
        padding=kernel.shape[0]//2, 
        dilation=1, 
        groups=1, 
        bias=False
    )
    
    conv.weight = torch.nn.Parameter(data=kernel_prep, requires_grad=False)
    
    images_conv = conv(images_spconv)
    return sparse_convert_spconv_to_scipy(images_conv)

def give_sparse_len(sparse_matrix):
    class obj_give_sparse_len:
#     class obj_give_sparse_len(type(sparse_matrix)):
        def __init__(self, sparse_matrix):
#             super().__init__(sparse_matrix)
            self.sp_matrix = sparse_matrix

        def __len__(self):
            return self.sp_matrix.shape[0]
        
        def __getitem__(self, idx):
            return self.sp_matrix[idx]
    
    return obj_give_sparse_len(sparse_matrix)

def batch_2D_sparse_convolution(images, kernel, image_shape, batch_size=None, num_batches=100, device='cpu'):
    images_w_len = give_sparse_len(images)
    
    return scipy.sparse.vstack([sparse_conv2D(
        images=batch, 
        kernel=kernel,
        image_shape=image_shape,
        device=device
    ) for batch in make_batches(images_w_len, batch_size=batch_size, num_batches=num_batches)])


##########################
##########################
##########################

def import_pth_model(path_pth, path_py):
    # Instantiate Model
    import importlib
    model_module = importlib.import_module(path_py)
    model = model_module.get_model(path_pth)
    model.eval();
    return model, model_module

def get_latents_swt(sfs, swt, device_model):
    sfs = torch.as_tensor(np.ascontiguousarray(sfs[None,...]), device=device_model, dtype=torch.float32)
    latents_swt = swt(sfs[None,...]).squeeze()
    latents_swt = latents_swt.reshape(latents_swt.shape[0], -1)
    return latents_swt


###########################
####### block stuff #######
###########################

def make_block_batches(
    frame_height=512,
    frame_width=1024,
    block_height=100,
    block_width=100,
    overlapping_width_Multiplier=2,
    outer_block_height=None,
    outer_block_width=None,
    clamp_outer_block_to_frame=True,
):     
    
    # block prep
    block_height_half = block_height//2
    block_width_half = block_width//2
    
    # inner block prep
    if outer_block_height is None:
        outer_block_height = block_height * 1.5
        print(f'Outer block height not specified. Using {outer_block_height}')
    if outer_block_width is None:
        outer_block_width = block_width * 1.5
        print(f'Outer block width not specified. Using {outer_block_width}')
        
    outer_block_height_half = outer_block_height//2
    outer_block_width_half = outer_block_width//2
    
    # find centers of blocks
#     n_blocks_x = np.ceil((frame_width*overlapping_width_Multiplier) / block_width).astype(np.int64)
    n_blocks_x = np.ceil(frame_width / (block_width - (block_width*overlapping_width_Multiplier))).astype(np.int64)

    centers_x = np.linspace(
        start=block_width_half,
        stop=frame_width - block_width_half,
        num=n_blocks_x,
        endpoint=True
    )

#     n_blocks_y = frame_height / block_height
    n_blocks_y = np.ceil(frame_height / (block_height - (block_height*overlapping_width_Multiplier))).astype(np.int64)
#     n_blocks_y = np.ceil(n_blocks_y*overlapping_width_Multiplier).astype(np.int64)
    centers_y = np.linspace(
        start=block_height_half,
        stop=frame_height - block_height_half,
        num=n_blocks_y,
        endpoint=True
    )
    
#     print(n_blocks_x)
#     print(centers_x)
    
    # make blocks
    blocks, outer_blocks = [], []
    for i_x in range(n_blocks_x):
        for i_y in range(n_blocks_y):
            blocks.append([
                list(np.int64([centers_y[i_y] - block_height_half , centers_y[i_y] + block_height_half])),
                list(np.int64([centers_x[i_x] - block_width_half , centers_x[i_x] + block_width_half]))
            ])
            
            outer_blocks.append([
                list(np.int64([centers_y[i_y] - outer_block_height_half , centers_y[i_y] + outer_block_height_half])),
                list(np.int64([centers_x[i_x] - outer_block_width_half , centers_x[i_x] + outer_block_width_half]))                
            ])
            
    # clamp outer block to limits of frame
    if clamp_outer_block_to_frame:
        for ii, outer_block in enumerate(outer_blocks):
            br_h = np.array(outer_block[0]) # block range height
            br_w = np.array(outer_block[1]) # block range width
            valid_h = (br_h>0) * (br_h<frame_height)
            valid_w = (br_w>0) * (br_w<frame_width)
            outer_blocks[ii] = [
                list( (br_h * valid_h) + (np.array([0, frame_height])*np.logical_not(valid_h)) ),
                list( (br_w * valid_w) + (np.array([0, frame_width])*np.logical_not(valid_w)) ),            
            ]
        
    return blocks, outer_blocks, (centers_y, centers_x)


def visualize_blocks(inner_blocks, outer_blocks, frame_height=512, frame_width=1024):
    im = np.zeros((frame_height, frame_width, 3))
    for block in inner_blocks:
        im[block[0][0]:block[0][1], block[1][0]:block[1][1], 0] += 0.2
    for block in outer_blocks:
        im[block[0][0]:block[0][1], block[1][0]:block[1][1], 1] += 0.2
    plt.figure()
    plt.imshow(im)


###########################
## distance matrix stuff ##
###########################

def make_session_similarity_matrix(idx_sessions, d_diff=1, d_same=0):
    arr = np.zeros((len(idx_sessions), len(idx_sessions)), dtype=np.int16)
    arr = np.zeros((len(idx_sessions), len(idx_sessions)), dtype=np.bool8)
    for ii in range(idx_sessions.max().astype(np.int64) + 1):
        arr += (idx_sessions==ii)[:,None] @ (idx_sessions==ii)[None,:]

    return arr*(d_same - d_diff) + d_diff
#     return arr

def toDense_fill(sparse_array, fill_value=1, make_fill_value_max=None):
    if make_fill_value_max:
        fill_value = sparse_array.max()
    arr = sparse.asCOO(sparse_array)
    arr.fill_value = fill_value
    return arr.todense()


###############
## EMBEDDING ##
###############

def embed_ROIs(
    outer_blocks,
    inner_blocks,
    ROIs_toUse,
    frame_height,
    frame_width,
    idx_roi_session,
    tsne,
    i_block,
    mask_power,
    max_n_neighbors,
    pref_use_NN_distances,
    latents,
    latents_swt,
    max_perplexity
):
    def get_ROI_idx_from_block_idx(block, spatial_footprints_flat, frame_height=frame_height, frame_width=frame_width):
        """
        This function:
         1. gets the indices of all the pixels in the block (return: idx_block)
         2. crops each spatial footprint image (size frame_height x frame_width) to just 
            the block pixels (return: sf_block_allSesh)
         3. finds the ROIs that are within the block and makes boolean and index output arrays listing them.
            Outputs:
             a. bool_ROI_inBlock_allSesh: a list where each entry is a session and each element is a boolean
                array listing whether that ROI is in the block
             b. idx_ROI_inBlock_allSesh: a list where each entry is a session and each element is the index
                of an ROI that is in the block
             c. boolCat_ROI_inBlock: an array listing which ROIs across all sessions that are in the block
             d. idxCat_ROI_inBlock: an array listing the indices of ROIs across all sessions that are in 
                the block
        """
        idx_block = np.reshape(
            np.ravel_multi_index(
                np.meshgrid(np.arange(block[0][0], block[0][1]), np.arange(block[1][0], block[1][1])),
                (frame_height, frame_width),
                order='C'),
            newshape=-1,
            order='F')

        sf_block_allSesh         = [sfs[:,idx_block] for sfs in spatial_footprints_flat]
        bool_ROI_inBlock_allSesh = [np.array(sfs.sum(1) > 0).squeeze() for sfs in sf_block_allSesh]
        idx_ROI_inBlock_allSesh  = [np.where(bool_ROI)[0] for bool_ROI in bool_ROI_inBlock_allSesh]

        boolCat_ROI_inBlock = np.concatenate(bool_ROI_inBlock_allSesh)
        idxCat_ROI_inBlock = np.where(boolCat_ROI_inBlock)[0]
        
        return idx_block, bool_ROI_inBlock_allSesh, idx_ROI_inBlock_allSesh, boolCat_ROI_inBlock, idxCat_ROI_inBlock, sf_block_allSesh
    
    idx_outer_block, bool_ROI_inOuterBlock_allSesh, idx_ROI_inOuterBlock_allSesh, boolCat_ROI_inOuterBlock, idxCat_ROI_inOuterBlock, sf_OuterBlock_allSesh = \
        get_ROI_idx_from_block_idx(
            block=outer_blocks[i_block], 
            spatial_footprints_flat=ROIs_toUse, 
            frame_height=frame_height, 
            frame_width=frame_width
        )
        
    idx_inner_block, bool_ROI_inInnerBlock_allSesh, idx_ROI_inInnerBlock_allSesh, boolCat_ROI_inInnerBlock, idxCat_ROI_inInnerBlock, sf_InnerBlock_allSesh = \
        get_ROI_idx_from_block_idx(
            block=inner_blocks[i_block], 
            spatial_footprints_flat=ROIs_toUse, 
            frame_height=frame_height, 
            frame_width=frame_width
        )

    
    sf_block_inROIs_allSesh      = [sfs[idx_ROI_inOuterBlock_allSesh[ii],:] for ii,sfs in enumerate(sf_OuterBlock_allSesh)]
    sf_block_flat        = scipy.sparse.vstack([sfs for sfs in sf_block_inROIs_allSesh])
    sf_block_flat_scaled = sf_block_flat.power(mask_power)
    sf_block_flat_scaled = sf_block_flat_scaled.multiply( 0.5 / sf_block_flat_scaled.sum(1)) # this scaling makes each ROI sum to 0.5, so the max distance between two ROIs is 1
    sf_block_flat_scaled = scipy.sparse.csr_matrix(sf_block_flat_scaled)
    
    n_neighbors = min(sf_block_flat_scaled.shape[0]-1, max_n_neighbors)
#     print(f'Using n_neighbors: {n_neighbors}')

    distances_IOU = sklearn.neighbors.NearestNeighbors(
        algorithm='auto',
        leaf_size=30, 
        metric='manhattan',
        p=1,
    #     metric_params=None, 
        n_jobs=-1
    ).fit(sf_block_flat_scaled).kneighbors_graph(
        sf_block_flat_scaled,
        n_neighbors=n_neighbors,
        mode='distance'
    )

#     distances_IOU[distances_IOU>0.99] = distances_IOU[distances_IOU>0.99]*100


    if pref_use_NN_distances:
#     if False:
        ltu = latents[idxCat_ROI_inOuterBlock]
        ltu = 1*(ltu / torch.sum(ltu, dim=1, keepdim=True))
        dist_latents_NN = torch.cdist(ltu, ltu, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary').numpy()

        ltu = latents_swt[idxCat_ROI_inOuterBlock]
        ltu = 1*(ltu / torch.sum(ltu, dim=1, keepdim=True))
        dist_latents_swt = torch.cdist(ltu, ltu, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary').numpy()
        
        dist_latents = dist_latents_NN * dist_latents_swt
#         dist_latents = dist_latents_NN 
        
#         ltu = latents_swt[idxCat_ROI_inOuterBlock]
#         ltu = 0.7071067811865476*(ltu / torch.sum(ltu, dim=1, keepdim=True))
#         cc_latents = np.corrcoef(ltu)
#         cc_latents[cc_latents<0] = 0
#         dist_latents = ((0+1) / (cc_latents**2)) -1
#         dist_latents = sim2dist(cc_latents,1,1)

        r = distances_IOU.tocoo().row
        c = distances_IOU.tocoo().col
        dist_latents_sparse = scipy.sparse.csr_matrix((dist_latents[r,c], (r,c)), shape=dist_latents.shape)

        distances_toUse = distances_IOU.power(1).multiply(dist_latents_sparse.power(1))
    else:
        distances_toUse = distances_IOU.power(1)
    
    
    distances_sessions = make_session_similarity_matrix(idx_roi_session[boolCat_ROI_inOuterBlock], d_diff=0, d_same=1)
    distances_sessions = distances_sessions * np.logical_not(np.eye(distances_sessions.shape[0]))
    r = distances_IOU.tocoo().row
    c = distances_IOU.tocoo().col
    distances_sessions_sparse = scipy.sparse.csr_matrix((distances_sessions[r,c], (r,c)), shape=distances_sessions.shape)

    
    distances_toUse[distances_IOU>1] = 1
    distances_toUse[distances_sessions_sparse.astype(np.bool8)] = 1
#     distances_toUse[distances_toUse>0.99] = 2
    
    tsne.n_neighbors = n_neighbors
    tsne.perplexity = min(max_perplexity, distances_toUse.shape[0]//4)
    # print(tsne.perplexity)
    
#     if pref_use_GPU:
#         embeddings = tsne.fit_transform(
#             X=sf_block_flat_scaled,
#             knn_graph=distances_toUse
#         )
#     else:
    # if tsne.method == 'exact':
    distances_toUse = toDense_fill(distances_toUse, 1)
        
    embeddings = tsne.fit_transform(
        X=distances_toUse,
#             X=distances_toUse.toarray(),
    )


    block_rois_oneBlock = {

        "idx_outer_block": idx_outer_block,
        "idx_inner_block": idx_inner_block,
        
        "bool_ROI_inOuterBlock_allSesh": bool_ROI_inOuterBlock_allSesh, # shape: (list len n_blocks (n_roi per block)). value: whether that roi is in the outer block
        "idx_ROI_inOuterBlock_allSesh" :  idx_ROI_inOuterBlock_allSesh, # shape: (list len n_blocks (n_roi in outer block)). value: indices of rois in the outer block
        "boolCat_ROI_inOuterBlock": boolCat_ROI_inOuterBlock, # shape: (n_roi across sessions). value: whether that roi is in the outer block
        "idxCat_ROI_inOuterBlock": idxCat_ROI_inOuterBlock, # shape: (n_roi in the outer block across all sessions). value: indices of the rois in the outer block
        
        "bool_ROI_inInnerBlock_allSesh": bool_ROI_inInnerBlock_allSesh, # shape: (list len n_blocks (n_roi per block)). value: whether that roi is in the inner block
        "idx_ROI_inInnerBlock_allSesh" :  idx_ROI_inInnerBlock_allSesh, # shape: (list len n_blocks (n_roi in inner block)). value: indices of rois in the inner block
        "boolCat_ROI_inInnerBlock": boolCat_ROI_inInnerBlock, # shape: (n_roi across sessions). value: whether that roi is in the inner block
        "idxCat_ROI_inInnerBlock": idxCat_ROI_inInnerBlock, # shape: (n_roi in the inner block across all sessions). value: indices of the rois in the inner block
        
        "n_neighbors": n_neighbors,
        "distances": distances_toUse,
        "tsne": tsne,
        "embeddings": embeddings,
    }
    
    return block_rois_oneBlock


################
## Clustering ##
################

def display_clustering_widget(
    embeddings, 
    min_samples, 
    max_samples=None,
    min_slider=0.01,
    max_slider=5, 
    start_slider=1.0,
    single_color=False,
    ):

    from ipywidgets import interact, widgets
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(embeddings[:,0], embeddings[:,1], s=15)
    # sc = ax.scatter(embeddings2[:,0], embeddings2[:,1], embeddings2[:,2], s=15)

    def get_val_counts(vals):
        vals_unique = np.unique(vals)
        vals_counts = np.zeros_like(vals, dtype=np.int64)
        for ii, val in enumerate(vals_unique):
            vals_counts[vals==val] = np.sum(vals==val)
        return vals_counts
    def update(eps = 1.0):
        eps_toUse = (eps/200)**2
        # Compute DBSCAN
        db = sklearn.cluster.DBSCAN(
            eps=eps_toUse,
            min_samples=min_samples, 
    #         metric='manhattan',
            metric_params=None, 
            algorithm='auto',
            leaf_size=30, 
            p=2, 
            n_jobs=-1
        ).fit(embeddings)

        labels = db.labels_
    #     labels = db.labels_ - db.labels_.min()
        if max_samples is not None:
            labels[get_val_counts(labels) > max_samples] = -1

        if single_color:
            cmap = rand_cmap(2, verbose=False)
        else:
            cmap = rand_cmap(len(np.unique(labels)), verbose=False)

        if len(np.unique(labels)) == 1:
            print('Eps value gives one big cluster. Adjust')
        else:
            sc.set_color(cmap(squeeze_integers(labels)+1))

            fig.canvas.draw_idle()
            ax.set_title(f'eps={round(eps_toUse,2)}, n_clusters={len(np.unique(labels))}')
            print(np.unique(labels))

    interact(update, eps=widgets.IntSlider(min=np.sqrt(min_slider)*200, max=np.sqrt(max_slider)*200, step=1, value=np.sqrt(start_slider)*200));


# def run_clustering_sweep(block_rois_raw, dbscan_objs, idx_roi_session):
#     def is_unique(vals):
#         is_unique = len(vals) == len(np.unique(vals))
#         return is_unique
#     def freq_of_values(vals):
#         u = np.unique(vals)
#         f = np.array([np.sum(vals==unique) for unique in u])
#         return np.array([f[u==val][0] for val in vals])

#     block_rois_multiEps = [copy.deepcopy(block) for block in block_rois_raw*len(dbscan_objs)] 

#     n_bse = n_blocks_single_eps = len(block_rois_raw)
#     for jj, dbscan in enumerate(tqdm(dbscan_objs)):
#         for ii, block in enumerate(block_rois_raw):
#             ## DBSCAN
#             db = copy.deepcopy(dbscan)
#             db.fit(block['embeddings']) # note that db.labels_==-1 means no cluster found
#             db.labels_[freq_of_values(db.labels_) < dbscan.min_samples] = -1 # fail safe because sometimes there are clusters of just 1 for some reason...

#             labels_unique = np.unique(db.labels_)
#             idxCat_ROI_inOuterBlock = block_rois_multiEps[jj*n_bse + ii]['idxCat_ROI_inOuterBlock']

#             block_rois_multiEps[jj*n_bse + ii]['db'] = db
#             block_rois_multiEps[jj*n_bse + ii]['labels_unique'] = labels_unique
#             block_rois_multiEps[jj*n_bse + ii]['cluster_sessions'] = [idx_roi_session[idxCat_ROI_inOuterBlock[db.labels_==label]] for label in labels_unique] # list of sessions a unique label derives from. Shape: [n_unique_labels][num of sessions where label is found]
#             block_rois_multiEps[jj*n_bse + ii]['cluster_sessions_isUnique'] = np.array([is_unique(sessions) for sessions in block_rois_multiEps[jj*n_bse + ii]['cluster_sessions']]) # boolean. Shape (n_labels_unique). Value whether all the labels derive from unique sessions.    
#             block_rois_multiEps[jj*n_bse + ii]['sizes_clusters'] = [(db.labels_==cid).sum() for cid in labels_unique]

#     return block_rois_multiEps

def run_clustering_sweep(block_rois_raw, cDBSCAN_class, idx_roi_session, min_samples=2):
    def is_unique(vals):
        return len(vals) == len(np.unique(vals))
    def freq_of_values(vals):
        u = np.unique(vals)
        f = np.array([np.sum(vals==unique) for unique in u])
        return np.array([f[u==val][0] for val in vals])

    block_rois_out = copy.deepcopy(block_rois_raw)

    for ii, block in enumerate(tqdm(block_rois_raw)):
        clusters_idx_unique, clusters_idx_unique_freq = cDBSCAN_class.fit(block['embeddings'])

        sizes_clusters_raw = np.array([len(idx) for idx in clusters_idx_unique], dtype=np.int64)
        # print(clusters_idx_unique)
        idx_tooSmall = np.where(sizes_clusters_raw < min_samples)[0]

        ## remove clusters with too few samples
        clusters_idx_unique = np.delete(clusters_idx_unique, idx_tooSmall)
        clusters_idx_unique_freq = np.delete(clusters_idx_unique_freq, idx_tooSmall)

        sizes_clusters = np.array([len(idx) for idx in clusters_idx_unique], dtype=np.int64)
        block_rois_out[ii]['clusters_idx_unique'] = [block['idxCat_ROI_inOuterBlock'][idx] for idx in clusters_idx_unique]
        block_rois_out[ii]['clusters_idx_unique_freq'] = clusters_idx_unique_freq
        block_rois_out[ii]['cluster_sessions'] = [idx_roi_session[idx] for idx in block_rois_out[ii]['clusters_idx_unique']]
        block_rois_out[ii]['sizes_clusters'] = sizes_clusters
        block_rois_out[ii]['cluster_sessions_isUnique'] = [is_unique(sessions) for sessions in block_rois_out[ii]['cluster_sessions']]

    return block_rois_out

    # block_rois_multiEps = [copy.deepcopy(block) for block in block_rois_raw*len(dbscan_objs)] 

    # n_bse = n_blocks_single_eps = len(block_rois_raw)
    # for jj, dbscan in enumerate(tqdm(dbscan_objs)):
    #     for ii, block in enumerate(block_rois_raw):
    #         ## DBSCAN
    #         db = copy.deepcopy(dbscan)
    #         db.fit(block['embeddings']) # note that db.labels_==-1 means no cluster found
    #         db.labels_[freq_of_values(db.labels_) < dbscan.min_samples] = -1 # fail safe because sometimes there are clusters of just 1 for some reason...

    #         labels_unique = np.unique(db.labels_)
    #         idxCat_ROI_inOuterBlock = block_rois_multiEps[jj*n_bse + ii]['idxCat_ROI_inOuterBlock']

    #         block_rois_multiEps[jj*n_bse + ii]['db'] = db
    #         block_rois_multiEps[jj*n_bse + ii]['labels_unique'] = labels_unique
    #         block_rois_multiEps[jj*n_bse + ii]['cluster_sessions'] = [idx_roi_session[idxCat_ROI_inOuterBlock[db.labels_==label]] for label in labels_unique] # list of sessions a unique label derives from. Shape: [n_unique_labels][num of sessions where label is found]
    #         block_rois_multiEps[jj*n_bse + ii]['cluster_sessions_isUnique'] = np.array([is_unique(sessions) for sessions in block_rois_multiEps[jj*n_bse + ii]['cluster_sessions']]) # boolean. Shape (n_labels_unique). Value whether all the labels derive from unique sessions.    
    #         block_rois_multiEps[jj*n_bse + ii]['sizes_clusters'] = [(db.labels_==cid).sum() for cid in labels_unique]

    # return block_rois_multiEps


def reshape_into_sparse(arr, idx, shape):
    idx = [t.reshape(-1) for t in np.meshgrid(idx, idx)]
    sp = scipy.sparse.csr_matrix((arr.reshape(-1), idx), shape=shape)
    return sp
## combine distances matrices into big distance matrix
def combine_distances_from_blocks(block_rois, n_roi_all):
    h = scipy.sparse.csr_matrix((n_roi_all, n_roi_all))
    for i_block, block in enumerate(tqdm(block_rois)):
        idx = block['idxCat_ROI_inOuterBlock']
        new = reshape_into_sparse(block['distances'], idx, shape=(n_roi_all, n_roi_all))
        # h = h + new
        h = h.maximum(new)
        
    return h