import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors

import gc

class ROI_graph:
    """
    Class for building similarity and distance graphs
     between ROIs based on their features
    """
    def __init__(
        self,
        device='cpu',
        spatialFootprint_maskPower=0.8,
        algorithm_nearestNeigbors_spatialFootprints='brute',
        n_neighbors_nearestNeighbors_spatialFootprints='full',
        **kwargs_nearestNeigbors_spatialFootprints
    ):
        self._device = device
        self._sf_maskPower = spatialFootprint_maskPower
        self._algo_sf = algorithm_nearestNeigbors_spatialFootprints
        self._nn_sf = n_neighbors_nearestNeighbors_spatialFootprints
        self._kwargs_sf = kwargs_nearestNeigbors_spatialFootprints

    def compute_similarity_graph(
        self,
        spatialFootprints,
        features_NN,
        features_SWT,
        ROI_session_bool,
    ):

        sf = scipy.sparse.vstack(spatialFootprints)
        sf = sf.power(self._sf_maskPower)
        sf = sf.multiply( 0.5 / sf.sum(1))
        sf = scipy.sparse.csr_matrix(sf)

        d_sf = sklearn.neighbors.NearestNeighbors(
            algorithm=self._algo_sf,
            metric='manhattan',
            p=1,
            n_jobs=-1,
            **self._kwargs_sf
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=self._nn_sf if self._nn_sf != 'full' else sf.shape[0],
            mode='distance'
        )
        del sf

        d_NN  = torch.cdist(features_NN.to(self._device),  features_NN.to(self._device),  p=2).cpu()
        d_SWT = torch.cdist(features_SWT.to(self._device), features_SWT.to(self._device), p=2).cpu()

        s_sf = 1 - d_sf.toarray()
        s_sf[s_sf < 0] = 0
        s_sf[range(s_sf.shape[0]), range(s_sf.shape[0])] = 0
        s_sf = torch.as_tensor(s_sf, dtype=torch.float32)
        del d_sf

        s_NN = 1 / (d_NN / d_NN.max())
        s_NN[s_NN < 0] = 0
        s_NN[range(s_NN.shape[0]), range(s_NN.shape[0])] = 0
        del d_NN

        s_SWT = 1 / (d_SWT / d_SWT.max())
        s_SWT[s_SWT < 0] = 0
        s_SWT[range(s_SWT.shape[0]), range(s_SWT.shape[0])] = 0
        del d_SWT

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