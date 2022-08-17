import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import sparse

import gc
import multiprocessing as mp
import time
import copy

from . import helpers


class ROI_graph:
    """
    Class for:
     1. Building similarity and distance graphs between ROIs
      based on their features.
     2. Generating potential clusters of ROIs using linkage 
      clustering.
     3. Building a similarity graph between clusters of ROIs.
     4. Computing silhouette scores for each potential cluster.

    To accelerate computation and reduce memory usage, some of the
     computations are performed on 'blocks' of the full field of
     view.
    
    RH 2022
    """
    def __init__(
        self,
        device='cpu',
        n_workers=1,
        spatialFootprint_maskPower=0.8,
        frame_height=512,
        frame_width=1024,
        block_height=100,
        block_width=100,
        overlapping_width_Multiplier=0.2,
        outer_block_height=None,
        outer_block_width=None,
        algorithm_nearestNeigbors_spatialFootprints='brute',
        n_neighbors_nearestNeighbors_spatialFootprints='full',
        locality=1,
        verbose=True,
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
            locality (float):
                Value to use as an exponent for the cluster similarity calculations
                self.s remains unchanged, but self.c is computed using self.s**locality
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

        self._locality = locality

        self._verbose = verbose

        self.n_workers = mp.cpu_count() if n_workers == -1 else n_workers

        self.frame_height = frame_height
        self.frame_width = frame_width

        self.blocks, self.outer_blocks, (self._centers_y, self._centers_x) = self._make_block_batches(
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=block_height,
            block_width=block_width,
            overlapping_width_Multiplier=overlapping_width_Multiplier,
            outer_block_height=outer_block_height,
            outer_block_width=outer_block_width,
            clamp_outer_block_to_frame=True,
        )


    def _compute_similarity_blockwise(
        self,
        spatialFootprints,
        features_NN,
        features_SWT,
        ROI_session_bool,
        linkage_methods=['single', 'complete', 'ward', 'average'],
        linkage_distances=[0.1, 0.2, 0.4, 0.8],
        min_cluster_size=2,
        max_cluster_size=None,
        batch_size_hashing=100,
    ):

        self._n_sessions = len(spatialFootprints)

        self._linkage_methods = linkage_methods
        self._linkage_distances = linkage_distances
        self._min_cluster_size = min_cluster_size
        self._max_cluster_size = max_cluster_size
        self._batch_size_hashing = batch_size_hashing

        self.sf_cat = scipy.sparse.vstack(spatialFootprints)
        n_roi = self.sf_cat.shape[0]

        # s_all = [scipy.sparse.lil_matrix((n_roi, n_roi))] * len(self.blocks)
        self.s = scipy.sparse.csr_matrix((n_roi, n_roi))
        s_empty = scipy.sparse.lil_matrix((n_roi, n_roi))
        self.d = scipy.sparse.csr_matrix((n_roi, n_roi))
        cluster_idx_all = []

        self.idxPixels_block = []
        for block in self.blocks:
            idx_tmp = np.zeros((self.frame_height, self.frame_width), dtype=np.bool8)
            idx_tmp[block[0][0]:block[0][1], block[1][0]:block[1][1]] = True
            idx_tmp = np.where(idx_tmp.reshape(-1))[0]
            self.idxPixels_block.append(idx_tmp)

        # for ii, block in tqdm(enumerate(self.blocks), total=len(self.blocks)):
        for ii, block in tqdm(enumerate(self.blocks), total=len(self.blocks)):
            idxROI_block = np.where(self.sf_cat[:, self.idxPixels_block[ii]].sum(1) > 0)[0]

            s_block = self._compute_ROI_similarity_graph(
                spatialFootprints=self.sf_cat[idxROI_block][:, self.idxPixels_block[ii]].power(self._sf_maskPower),
                features_NN=features_NN[idxROI_block],
                features_SWT=features_SWT[idxROI_block],
                ROI_session_bool=ROI_session_bool[idxROI_block],
            )
            s_block = torch.maximum(s_block, s_block.T)  # force symmetry
            idx = np.meshgrid(idxROI_block, idxROI_block)
            s_tmp = copy.copy(s_empty)
            s_tmp[idx[0], idx[1]] = s_block
            self.s = self.s.maximum(s_tmp)

            d_block = 1 / s_block
            d_block[range(d_block.shape[0]), range(d_block.shape[0])] = 0
            d_block[torch.isinf(d_block).type(torch.bool)] = 1e10  ## convert inf

            ## if d_block.shape[0] is 1 or 0, then there aren't enough samples in the block to find clusters
            if d_block.shape[0] > 1:
                cluster_idx_block__idxBlock, cluster_freq_block =  self._compute_linkage_clusters(d_block.numpy())
                cluster_idx_block = [idxROI_block[idx] for idx in cluster_idx_block__idxBlock]

                cluster_idx_all += cluster_idx_block
        # return cluster_idx_all

        u, idx, c = np.unique(
            ar=np.array([hash(tuple(vec)) for vec in cluster_idx_all]),
            return_index=True,
            return_counts=True,
        )

        # return u, idx, c
        self.cluster_idx = np.array(cluster_idx_all, dtype=np.int64)[idx]
        self.cluster_bool = scipy.sparse.vstack([scipy.sparse.csr_matrix(helpers.idx2bool(cid, length=self.s.shape[0])) for cid in self.cluster_idx])

        # return self.cluster_idx

        # print([sf_cat[idx].sum(0) for idx in self.cluster_idx])


    def _compute_ROI_similarity_graph(
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

        d_sf = sklearn.neighbors.NearestNeighbors(
            algorithm=self._algo_sf,
            metric='manhattan',
            p=1,
            # n_jobs=self.n_workers,
            n_jobs=-1,
            **self._kwargs_sf
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=sf.shape[0],
            mode='distance'
        )


        s_sf = 1 - d_sf.toarray()
        s_sf[s_sf < 0] = 0
        s_sf[range(s_sf.shape[0]), range(s_sf.shape[0])] = 0
        s_sf = torch.as_tensor(s_sf, dtype=torch.float32)

        d_NN  = torch.cdist(features_NN.to(self._device),  features_NN.to(self._device),  p=2).cpu()
        s_NN = 1 / (d_NN / d_NN.max())
        s_NN[s_NN < 0] = 0
        s_NN[range(s_NN.shape[0]), range(s_NN.shape[0])] = 0

        d_SWT = torch.cdist(features_SWT.to(self._device), features_SWT.to(self._device), p=2).cpu()
        s_SWT = 1 / (d_SWT / d_SWT.max())
        s_SWT[s_SWT < 0] = 0
        s_SWT[range(s_SWT.shape[0]), range(s_SWT.shape[0])] = 0

        session_bool = torch.as_tensor(ROI_session_bool, device='cpu', dtype=torch.float32)
        s_sesh = torch.logical_not((session_bool @ session_bool.T).type(torch.bool))

        s_conj = s_sf * s_NN * s_SWT * s_sesh

        return s_conj


    def _compute_linkage_clusters(
        self, 
        d,
    ):
        """
        Linkage clustering of the similarity graph
        """

        verbose = False

        d_sq = scipy.spatial.distance.squareform(d)
        print(f'Starting: computing linkage') if verbose else None
        # self.links = helpers.merge_dicts(helpers.simple_multiprocessing(helper_compute_linkage, (self._linkage_methods, [d_sq]*len(self._linkage_methods)), workers=self.n_workers))
        self.links = helpers.merge_dicts([helper_compute_linkage(method, d_sq) for method in self._linkage_methods])
        print(f'Completed: computing linkage') if verbose else None

        print(f'Starting: clustering') if verbose else None
        cluster_bool_all = []
        # print(self.links)
        for ii, t in enumerate(self._linkage_distances):
            [cluster_bool_all.append(self._helper_get_boolean_clusters_from_linkages(self.links[method], t=t, criterion='distance')) for method in self._linkage_methods]
            # [cluster_bool_all.append(scipy.sparse.csr_matrix(labels_to_bool(scipy.cluster.hierarchy.fcluster(self.links[method], t=t, criterion='distance')))) for method in self._linkage_methods]
            # print([self._helper_get_boolean_clusters_from_linkages(self.links[method], t=t, criterion='distance') for method in self._linkage_methods])
        
        # print(cluster_bool_all)
        cluster_bool_all = scipy.sparse.vstack(cluster_bool_all)
        print(f'Completed: clustering') if verbose else None

        print(f'Starting: filtering clusters by size removing redundant clusters') if verbose else None
        max_cluster_size = self._n_sessions if self._max_cluster_size is None else self._max_cluster_size
        nROIperCluster = np.array(cluster_bool_all.sum(1)).squeeze()
        idx_clusters_inRange = np.where( (nROIperCluster >= self._min_cluster_size) & (nROIperCluster <= max_cluster_size) )[0]
        
        cluster_idx, cluster_freq = self._get_unique_clusters(cluster_bool_all[idx_clusters_inRange])

        return cluster_idx, cluster_freq

    def _helper_get_boolean_clusters_from_linkages(self, links, t, criterion='distance'):
        ## if there aren't enough links to make a cluster, return an empty matrix
        # if links.ndim == 1:
        # # if False:
        #     return []
        # else:
        return scipy.sparse.csr_matrix(labels_to_bool(scipy.cluster.hierarchy.fcluster(links, t=t, criterion=criterion)))



    def _get_unique_clusters(
        self,
        cluster_bool,
    ):
        clusterHashes = np.concatenate([
            hash_matrix(batch) for batch in helpers.make_batches(cluster_bool, batch_size=self._batch_size_hashing, length=cluster_bool.shape[0])
        ], axis=0)
        
        u, idx, c = np.unique(
            ar=clusterHashes,
            return_index=True,
            return_counts=True,
        )

        cluster_idx = [torch.LongTensor(vec.indices) for vec in cluster_bool[idx]]
        cluster_freq = c

        return cluster_idx, cluster_freq

    
    def _compute_cluster_similarity_graph(
        self, 
        cluster_similarity_reduction_intra='mean',
        cluster_similarity_reduction_inter='max',
        cluster_silhouette_reduction_intra='mean',
        cluster_silhouette_reduction_inter='max',
        n_workers=None,
        ):

        self._cluster_similarity_reduction_intra_method = cluster_similarity_reduction_intra
        self._cluster_similarity_reduction_inter_method = cluster_similarity_reduction_inter

        self._cluster_silhouette_reduction_intra_method = cluster_silhouette_reduction_intra
        self._cluster_silhouette_reduction_inter_method = cluster_silhouette_reduction_inter

        if n_workers is None:
            n_workers = mp.cpu_count()

        self.idxPixels_block = []
        for block in self.blocks:
            idx_tmp = np.zeros((self.frame_height, self.frame_width), dtype=np.bool8)
            idx_tmp[block[0][0]:block[0][1], block[1][0]:block[1][1]] = True
            idx_tmp = np.where(idx_tmp.reshape(-1))[0]
            self.idxPixels_block.append(idx_tmp)

        self.n_clusters = len(self.cluster_idx)

        ## make a sparse matrix of the spatial footprints of the sum of each cluster
        print('Starting: Making cluster spatial footprints') if self._verbose else None
        self.spatialFootprints_coo = sparse.COO(self.sf_cat)
        self.clusterBool_coo = sparse.COO(self.cluster_bool)
        batch_size = int(max(1e8 // self.spatialFootprints_coo.shape[0], 1000))

        self.sf_clusters = sparse.concatenate(
            helpers.simple_multithreading(
                self._helper_make_sfClusters,
                [helpers.make_batches(self.clusterBool_coo[:,:,None], batch_size=batch_size)],
                workers=n_workers
            )
        ).tocsr()
        print('Completed: Making cluster spatial footprints') if self._verbose else None


        print('Starting: Computing cluster similarities') if self._verbose else None
        self.c_sim = scipy.sparse.lil_matrix((self.n_clusters, self.n_clusters))
        self.s_local = sparse.COO(self.s.power(self._locality))
        self._idxClusters_block = [np.where(self.sf_clusters[:, idx_pixels].sum(1) > 0)[0] for idx_pixels in self.idxPixels_block]  ## find indices of the clusters that have at least one non-zero pixel in the block

        helpers.simple_multithreading(
            self._helper_compute_cluster_similarity_batch,
            [np.arange(len(self.blocks))],
            workers=n_workers
        )
        print('Completed: Computing cluster similarities') if self._verbose else None


        print('Starting: Computing modified cluster silhouettes') if self._verbose else None
        
        self._cluster_bool_coo = sparse.COO(self.cluster_bool)

        self._cluster_bool_inv = sparse.COO(self.cluster_bool)
        self._cluster_bool_inv.data = self._cluster_bool_inv.data < True
        self._cluster_bool_inv.fill_value = True

        self.c_sil = np.array([self._helper_cluster_silhouette(
            idx, 
            method_in=self._cluster_silhouette_reduction_intra_method,
            method_out=self._cluster_silhouette_reduction_inter_method,
        ) for idx in tqdm(range(self.n_clusters))])

        print('Completed: Computing modified cluster silhouettes') if self._verbose else None



    def _helper_make_sfClusters(
        self,
        cb_s_batch
    ):
        return (self.spatialFootprints_coo[None,:,:] * cb_s_batch).sum(axis=1)


    def _helper_compute_cluster_similarity_batch(
        self,
        i_block, 
    ):
        cBool = sparse.COO(self.cluster_bool[self._idxClusters_block[i_block]])
        sizes_clusters = cBool.sum(1)

        cs_inter = (self.s_local[None,None,:,:] * cBool[:, None, :, None]) * cBool[None, :, None, :]  ## arranges similarities between every roi ACROSS every pair of clusters. shape (n_clusters, n_clusters, n_ROI, n_ROI)
        c_block = reduction_inter(
            x=cs_inter, 
            sizes_clusters=sizes_clusters, 
            method=self._cluster_similarity_reduction_inter_method,
        ).todense()  ## compute the reduction of the cs_inter array along the ROI dimensions

        cs_intra = (self.s_local[None,:,:] * cBool[:, :, None]) * cBool[:, None, :]  ## arranges similarities between every roi WITHIN each cluster. shape (n_clusters, n_ROI, n_ROI)
        c_block[range(c_block.shape[0]), range(c_block.shape[0])] = reduction_intra(
            cs_intra, 
            sizes_clusters=sizes_clusters,
            method=self._cluster_similarity_reduction_intra_method,
        ).todense()  ## compute the reduction of the cs_intra array along the ROI dimensions

        c_block = np.maximum(c_block, c_block.T)  # force symmetry

        idx = np.meshgrid(self._idxClusters_block[i_block], self._idxClusters_block[i_block])
        self.c_sim[idx[0], idx[1]] = c_block
        return c_block

    def _helper_cluster_silhouette(
        self,
        idx,
        method_in='mean',
        method_out='max',
    ):
        assert method_in in ['mean', 'min'], 'method_in must be mean or min'
        assert method_out in ['max'], 'method_out must be max'

        idx_in = self._cluster_bool_coo[idx].nonzero()[0]
        s_single = self.s_local[idx_in]
        if method_in == 'mean':
            cs_in = s_single[:, idx_in].mean()
        elif method_in == 'min':
            cs_in = s_single[:, idx_in].min()
        cs_out = (s_single * self._cluster_bool_inv[idx]).max()
        # return cs_in / cs_out
        return (cs_in - cs_out) / np.maximum(cs_in, cs_out)


###########################
####### block stuff #######
###########################

    def _make_block_batches(
        self,
        frame_height=512,
        frame_width=1024,
        block_height=100, 
        block_width=100,
        overlapping_width_Multiplier=0.2,
        outer_block_height=None,
        outer_block_width=None,
        clamp_outer_block_to_frame=True,
    ):     
        
        # block prep
        block_height_half = block_height//2
        block_width_half = block_width//2
        
        # inner block prep
        if outer_block_height is None:
            outer_block_height = block_height * 1.0
            print(f'Outer block height not specified. Using {outer_block_height}')
        if outer_block_width is None:
            outer_block_width = block_width * 1.0
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
        self,
    ):
        im = np.zeros((self.frame_height, self.frame_width, 3))
        for block in self.blocks:
            im[block[0][0]:block[0][1], block[1][0]:block[1][1], 0] += 0.2
        for block in self.outer_blocks:
            im[block[0][0]:block[0][1], block[1][0]:block[1][1], 1] += 0.2
        plt.figure()
        plt.imshow(im)



def labels_to_bool(labels):
#     return {label: np.where(labels==label)[0] for label in np.unique(labels)}
    return np.array([labels==label for label in np.unique(labels)])

def helper_compute_linkage(method, d_sq):
    # print(f'computing method: {method}')
    if len(d_sq) > 0:
        return {method : scipy.cluster.hierarchy.linkage(d_sq, method=method)}
    else:
        return {method : np.array([])}

def hash_matrix(x):
    y = np.array(np.packbits(x.todense(), axis=1))
    return np.array([hash(tuple(vec)) for vec in y])

def reduction_inter(x, sizes_clusters, method='max'):
    if method == 'max':
        return x.max(axis=(2,3))
    elif method == 'mean':
        return x.sum(axis=(2,3)) / (sizes_clusters[:,None] * sizes_clusters[None,:])

def reduction_intra(x, sizes_clusters, method='min'):
    if method == 'min':
        x.fill_value = np.inf
        return x.min(axis=(1,2))
    elif method == 'mean':
        return x.sum(axis=(1,2)) / (sizes_clusters * (sizes_clusters-1))
