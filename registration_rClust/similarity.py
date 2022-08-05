import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc
import multiprocessing as mp

from . import helpers


class ROI_graph:
    """
    Class for building similarity and distance graphs
     between ROIs based on their features, and for generating
     potential clusters of ROIs.
    These methods can be improved in their memory handling 
     by performing them over each block of the FOV.
    RH 2022
    """
    def __init__(
        self,
        device='cpu',
        n_workers=1,
        spatialFootprint_maskPower=0.8,
        algorithm_nearestNeigbors_spatialFootprints='brute',
        n_neighbors_nearestNeighbors_spatialFootprints='full',
        **kwargs_nearestNeigbors_spatialFootprints
    ):
        """
        Initialize the class.

        Args:
            device (str):
                The device to use for the computations.
            n_workers (int):
                The number of workers to use for the computations.
                Set to -1 to use all available cpu cores.
                Used for spatial footprint manahattan distance computation,
                 computing hashes of cluster idx, and computing linkages.
            spatialFootprint_maskPower (float):
                The power to use for the spatial footprint mask. Lower
                 values will make masks more binary looking for distance
                 computation.
            algorithm_nearestNeigbors_spatialFootprints (str):
                The algorithm to use for the nearest neighbors computation.
                See sklearn.neighbors.NearestNeighbors for more information.
            n_neighbors_nearestNeighbors_spatialFootprints (int or str):
                The number of neighbors to use for the nearest neighbors.
                Set to 'full' to use all available neighbors.
            **kwargs_nearestNeigbors_spatialFootprints (dict):
                The keyword arguments to use for the nearest neighbors.
                Optional.
                See sklearn.neighbors.NearestNeighbors for more information.
        """
        self._device = device
        self._sf_maskPower = spatialFootprint_maskPower
        self._algo_sf = algorithm_nearestNeigbors_spatialFootprints
        self._nn_sf = n_neighbors_nearestNeighbors_spatialFootprints
        self._kwargs_sf = kwargs_nearestNeigbors_spatialFootprints

        self.n_workers = mp.cpu_count() if n_workers == -1 else n_workers


    def compute_similarity_graph(
        self,
        spatialFootprints,
        features_NN,
        features_SWT,
        ROI_session_bool,
    ):
        """
        Compute the similarity matrix between ROIs based on the 
         conjunction of the similarity matrices for different modes
         (like the NN embedding, the SWT embedding, and the spatial footprint
         overlap).

        Args:
            spatialFootprints (list of scipy.sparse.csr_matrix):
                The spatial footprints of the ROIs.
                list of shape (n_ROIs for each session, FOV height * FOV width)
            features_NN (torch.Tensor):
                The output latent embeddings of the NN model.
                shape (n_ROIs total, n_features)
            features_SWT (torch.Tensor):
                The output latent embeddings of the SWT model.
                shape (n_ROIs total, n_features)
            ROI_session_bool (np.ndarray):
                The boolean matrix indicating which ROIs belong to which session.
                shape (n_ROIs total, n_sessions)
        """

        sf = scipy.sparse.vstack(spatialFootprints)
        sf = sf.power(self._sf_maskPower)
        sf = sf.multiply( 0.5 / sf.sum(1))
        sf = scipy.sparse.csr_matrix(sf)

        self.n_roi = sf.shape[0]
        self.n_sessions = len(spatialFootprints)

        d_sf = sklearn.neighbors.NearestNeighbors(
            algorithm=self._algo_sf,
            metric='manhattan',
            p=1,
            n_jobs=self.n_workers,
            **self._kwargs_sf
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=self._nn_sf if self._nn_sf != 'full' else sf.shape[0],
            mode='distance'
        )
        del sf


        s_sf = 1 - d_sf.toarray()
        del d_sf
        s_sf[s_sf < 0] = 0
        s_sf[range(s_sf.shape[0]), range(s_sf.shape[0])] = 0
        s_sf = torch.as_tensor(s_sf, dtype=torch.float32)

        d_NN  = torch.cdist(features_NN.to(self._device),  features_NN.to(self._device),  p=2).cpu()
        s_NN = 1 / (d_NN / d_NN.max())
        del d_NN
        s_NN[s_NN < 0] = 0
        s_NN[range(s_NN.shape[0]), range(s_NN.shape[0])] = 0

        d_SWT = torch.cdist(features_SWT.to(self._device), features_SWT.to(self._device), p=2).cpu()
        s_SWT = 1 / (d_SWT / d_SWT.max())
        del d_SWT
        s_SWT[s_SWT < 0] = 0
        s_SWT[range(s_SWT.shape[0]), range(s_SWT.shape[0])] = 0

        session_bool = torch.as_tensor(ROI_session_bool, device='cpu', dtype=torch.float32)
        s_sesh = torch.logical_not((session_bool @ session_bool.T).type(torch.bool))

        self.s_conj = s_sf * s_NN * s_SWT * s_sesh
        del s_sf, s_NN, s_SWT, s_sesh
        
        self.d_conj = 1 / self.s_conj
        self.d_conj[torch.isinf(self.d_conj).type(torch.bool)] = 10000  ## convert inf
        self.d_conj = torch.maximum( self.d_conj, self.d_conj.T )  ## force symmetry
        self.d_conj[torch.arange(self.d_conj.shape[0]), torch.arange(self.d_conj.shape[0])] = 0  ## set diagonal to 0
        self.d_conj = scipy.sparse.csr_matrix(self.d_conj.numpy())  ## sparsen

        gc.collect()

        return self.s_conj, self.d_conj

    
    def linkage_clustering(
        self, 
        linkage_methods=['single', 'complete', 'ward', 'average'],
        linkage_distances=[0.1, 0.2, 0.4, 0.8],
        min_cluster_size=2,
        max_cluster_size=None,
        batch_size=100,
        verbose=True,
    ):
        """
        Linkage clustering of the similarity graph
        """

        d_sq = scipy.spatial.distance.squareform(self.d_conj.toarray())
        print(f'Starting: computing linkage') if verbose else None
        self.links = helpers.merge_dicts(helpers.simple_multiprocessing(helper_compute_linkage, (linkage_methods, [d_sq]*len(linkage_methods)), workers=self.n_workers))
        print(f'Completed: computing linkage') if verbose else None

        print(f'Starting: clustering') if verbose else None
        cluster_bool_all = []
        for ii, t in tqdm(enumerate(linkage_distances)):
            [cluster_bool_all.append(scipy.sparse.csr_matrix(labels_to_bool(scipy.cluster.hierarchy.fcluster(self.links[method], t=t, criterion='distance')))) for method in linkage_methods]
        cluster_bool_all = scipy.sparse.vstack(cluster_bool_all)
        print(f'Completed: clustering') if verbose else None

        print(f'Starting: filtering clusters by size removing redundant clusters') if verbose else None
        max_cluster_size = self.n_sessions if max_cluster_size is None else max_cluster_size
        nROIperCluster = np.array(cluster_bool_all.sum(1)).squeeze()
        idx_clusters_inRange = np.where( (nROIperCluster >= min_cluster_size) & (nROIperCluster <= max_cluster_size) )[0]
        
        clusterHashes = np.concatenate(helpers.simple_multiprocessing(
            hash_matrix, 
            [helpers.make_batches(cluster_bool_all[idx_clusters_inRange], batch_size=batch_size, length=len(idx_clusters_inRange))], 
            workers=self.n_workers
        ), axis=0)
        
        u, idx, c = np.unique(
            ar=clusterHashes,
            return_index=True,
            return_counts=True,
        )

        # cluster_idx, cluster_idx_freq = np.unique(cluster_idx, axis=0, return_counts=True)

        self.cluster_idx = [torch.LongTensor(vec.indices) for vec in cluster_bool_all[idx_clusters_inRange][idx]]
        self.cluster_bool_freq = c
        print(f'Completed: filtering clusters by size removing redundant clusters') if verbose else None

        return self.cluster_bool


    def _cluster_similarity_score(
        self,
        s,
        cluster_bool,
        locality=1,
        method_in='mean',
        method_out='max',
    ):
        """
        Function to compute the aggregated similarity / dispersion score 
        of clusters.
        Here, the score measures the similarity between samples within
        a cluster and between samples within a cluster and all other samples.
        To compute a true silhouette score, use:
        method_in='mean' and method_out='max'.
        For a score similar to complete linkage, use:
        method_in='min' and method_out='max'.

        RH 2022

        Args:
            s (torch.Tensor, dtype float):
                The similarity matrix.
                shape: (n_samples, n_samples)
            cluster_bool (aka 'h') (scipy.sparse.csr_matrix, dtype bool):
                The boolean matrix indicating which samples are in each cluster.
                shape: (n_clusters, n_samples)
            locality (float):
                The exponent applied to the similarity matrix.
                Higher values make the score more dependent on local 
                similarity. 
                Setting method_out to 'mean' and using a high locality 
                value can result in something similar to a silhouette
                score.
            method_in (str):
                The method used to compute the within-cluster similarity.
                Must be one of "mean", "max", "min".
            method_out (str):
                The method used to compute the between-cluster similarity.
                Must be one of "mean", "max", "min".
        """

        self.cluster_idx = [torch.as_tensor(vec.indices, dtype=torch.int32, device=s.device) for vec in cluster_bool]

        n_clusters = cluster_bool.shape[0]
        n_samples = cluster_bool.shape[1]
        
        DEVICE = s.device
        s_tu = (s**locality).type(torch.float32)
        # h_tu = helpers.scipy_sparse_to_torch_coo(cluster_bool).to(DEVICE)

        # print(h_tu)

    
        ii_normFactor = lambda i   : i * (i-1)
        ij_normFactor = lambda i,j : i * j

        yy, xx = torch.meshgrid(torch.arange(n_clusters), torch.arange(n_clusters), indexing='ij')
        yyf, xxf = yy.reshape(-1), xx.reshape(-1)

        # sizes_clusters = h_tu.sum(1)
        sizes_clusters = [len(cIdx) for cIdx in self.cluster_idx]

        # print(sizes_clusters)
        # return sizes_clusters

        # if method_in=='mean' and method_out=='mean':
        #     s_tu[torch.arange(n_samples).to(DEVICE), torch.arange(n_samples).to(DEVICE)] = 0
        #     c = torch.einsum('ab, ac, bd -> cd', s_tu, h_tu, h_tu)  /  \
        #         ( (torch.eye(n_clusters).to(DEVICE) * ii_normFactor(sizes_clusters)) + ((1-torch.eye(n_clusters).to(DEVICE)) * (sizes_clusters[None,:] * sizes_clusters[:,None])) )
        #     return c


        # if h_tu.dtype != torch.bool:
        #     raise ValueError(f'h must be a boolean tensor. Got {h_tu.dtype}')

        # s_tu[torch.arange(n_samples).to(DEVICE), torch.arange(n_samples).to(DEVICE)] = torch.nan
        s_tu[range(n_samples), range(n_samples)] = torch.nan
        
        # return s_tu

        if method_in == 'mean':
            fn_mi = torch.nanmean
        elif method_in == 'max':
            fn_mi = helpers.nanmax
        elif method_in == 'min':
            fn_mi = helpers.nanmin
        else:
            raise ValueError('method_in must be one of "mean", "max", "min".')

        if method_out == 'mean':
            fn_mo = torch.nanmean
        elif method_out == 'max':
            fn_mo = helpers.nanmax
        elif method_out == 'min':
            fn_mo = helpers.nanmin
        else:
            raise ValueError('method_out must be one of "mean", "max", "min".')

        c = torch.as_tensor([fn_mo(s_tu[self.cluster_idx[ii]][:, self.cluster_idx[jj]]) for ii,jj in tqdm(zip(yyf, xxf), total=len(yyf))], device=DEVICE).reshape(n_clusters, n_clusters)
        c[torch.eye(n_clusters, dtype=torch.bool)] = torch.as_tensor([fn_mi(s_tu[self.cluster_idx[ii]][:, self.cluster_idx[ii]]) for ii in range(n_clusters)], device=DEVICE)
        
        # c = torch.as_tensor([fn_mo(s_tu[h_tu[ii]][:, h_tu[jj]]) for ii,jj in tqdm(zip(yyf, xxf), total=len(yyf))], device=DEVICE).reshape(n_clusters, n_clusters)
        # c[torch.eye(n_clusters, dtype=torch.bool)] = torch.as_tensor([fn_mi(s_tu[h_tu[ii]][:, h_tu[ii]]) for ii in range(n_clusters)], device=DEVICE)

        return c



def labels_to_bool(labels):
#     return {label: np.where(labels==label)[0] for label in np.unique(labels)}
    return np.array([labels==label for label in np.unique(labels)])

def helper_compute_linkage(method, d_sq):
    print(f'computing method: {method}')
    return {method : scipy.cluster.hierarchy.linkage(d_sq, method=method)}

def hash_matrix(x):
    y = np.array(np.packbits(x.todense(), axis=1))
    return np.array([hash(tuple(vec)) for vec in y])




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


def visualize_blocks(
    inner_blocks, 
    outer_blocks, 
    frame_height=512, 
    frame_width=1024
):
    im = np.zeros((frame_height, frame_width, 3))
    for block in inner_blocks:
        im[block[0][0]:block[0][1], block[1][0]:block[1][1], 0] += 0.2
    for block in outer_blocks:
        im[block[0][0]:block[0][1], block[1][0]:block[1][1], 1] += 0.2
    plt.figure()
    plt.imshow(im)