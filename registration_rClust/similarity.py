import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm import tqdm

import gc
import multiprocessing as mp

from . import helpers


class ROI_graph:
    """
    Class for building similarity and distance graphs
     between ROIs based on their features
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

        if n_workers == -1:
            self.n_workers = mp.cpu_count()


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
        self._cluster_idx_all = []
        for ii, t in tqdm(enumerate(linkage_distances)):
            [self._cluster_idx_all.append(scipy.sparse.csr_matrix(labels_to_bool(scipy.cluster.hierarchy.fcluster(self.links[method], t=t, criterion='distance')))) for method in linkage_methods]
        self._cluster_idx_all = scipy.sparse.vstack(self._cluster_idx_all)
        print(f'Completed: clustering') if verbose else None

    
    def find_unique_clusters(self,):
        clusterHashes = np.concatenate(helpers.simple_multiprocessing(
            hash_matrix, 
            [helpers.make_batches(self._cluster_idx_all, batch_size=100, length=self._cluster_idx_all.shape[0])], 
            workers=self.n_workers
        ), axis=0)

        # clusterHashes_block = [hash(tuple(vec)) for vec in cluster_idx]
        # u, idx, c = np.unique(
        #     ar=clusterHashes_block,
        #     return_index=True,
        #     return_counts=True,
        # )

        u, idx, c = np.unique(
            ar=clusterHashes,
            return_index=True,
            return_counts=True,
        )

        # cluster_idx, cluster_idx_freq = np.unique(cluster_idx, axis=0, return_counts=True)

        self.cluster_idx = self._cluster_idx_all[idx]
        self.cluster_idx_freq = c

        return self.cluster_idx


def labels_to_bool(labels):
#     return {label: np.where(labels==label)[0] for label in np.unique(labels)}
    return np.array([labels==label for label in np.unique(labels)])

def helper_compute_linkage(method, d_sq):
    print(f'computing method: {method}')
    return {method : scipy.cluster.hierarchy.linkage(d_sq, method=method)}

def hash_matrix(x):
    y = np.array(np.packbits(x.todense(), axis=1))
    return np.array([hash(tuple(vec)) for vec in y])