import gc

import torch
import torch_sparse as ts
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt

from . import helpers

class Cluster_Assigner:
    """
    Class to perform constrained cluster assignment.
    This method takes in putative clusters, a cluster similarity matrix,
     and a vector of cluster 'scores' describing how valuable each
     cluster is. It then attempts to find an optimal combination of
     clusters. This is done by minimizing inclusion of similar clusters
     and maximizing inclusion of clusters with high scores.
    The cluster similarity matrix and score vector can be made using 
     the cluster_dispersion_score and cluster_silhouette_score functions
     respectively. The cluster membership matrix showing putative clusters
     can be made by by doing something like sweeping over linkage distances.

    RH 2022
    """
    def __init__(
        self,
        c,
        h,
        w=None,
        m_init=None,
        device='cpu',
        optimizer_partial=None,
        scheduler_partial=None,
        dmCEL_temp=1,
        dmCEL_sigSlope=2,
        dmCEL_sigCenter=0.5,
        dmCEL_penalty=1,
        sampleWeight_softplusKwargs={'beta': 500, 'threshold': 50},
        sampleWeight_penalty=1e1,
        fracWeighted_goalFrac=1.0,
        fracWeighted_sigSlope=2,
        fracWeighted_sigCenter=0.5,
        fracWeight_penalty=1e0,
        maskL1_penalty=0e-3,
        tol_convergence=1e-2,
        window_convergence=100,
        freqCheck_convergence=100,
        verbose=True,
    ):
        """
        Args:
            c (scipy.sparse.csr_matrix, dtype float):
                The cluster similarity matrix.
                shape: (n_clusters, n_clusters)
            h (scipy.sparse.csr_matrix, dtype bool):
                The cluster membership matrix.
                shape: (n_samples, n_clusters)
            w (torch.Tensor, dtype float):
                The cluster score vector.
                shape: (n_clusters)
                Weighs how much to value the inclusion of each cluster
                 relative to the total fracWeight_penalty.
                If None, the clusters are weighted equally.
            m_init (torch.Tensor, dtype float):
                The initial cluster inclusion vector.
                shape: (n_clusters)
                If None, then vector is initialized as small random values.
            device (str):
                The device to use for the computation. ('cpu', 'cuda', etc.)
            optimizer_partial (torch.optim.Optimizer):
                A torch optimizer with all but the parameters initialized.
                If None, then the optimizer is initialized with Adam and 
                 some default parameters.
                This can be made using:
                 functools.partial(torch.optim.Adam, lr=x, betas=(x,x)).
            scheduler (torch.optim.lr_scheduler):
                A torch learning rate scheduler.
            dmCEL_temp (float):
                The temperature used in the cross entropy loss of the
                 interaction matrix.
            dmCEL_sigSlope (float):
                The slope of the sigmoid used to constrain and activate the
                 m vector. Higher values result in more forced certainty, 
                 but less stability.
                Recommend staying as low as possible, around 1-5.
            dmCEL_sigCenter (float):
                The center of the sigmoid described above.
                Recommend keeping to 0.5.
            dmCEL_penalty (float):
                The penalty applied to the cross entropy loss of the 
                 interaction matrix. Best to keep this to 1 and change the
                 other penalties.
            sampleWeight_softplusKwargs (dict):
                The kwargs passed to the softplus function used to penalize
                 when samples are included more than once. 
                Recommend keeping this around default values.
            sampleWeight_penalty (float):
                The penalty applied when samples are included more than once.
            fracWeighted_goalFrac (float):
                The goal fraction of samples to be included in the output.
                Recommend keeping this to 1 for most applications.
            fracWeighted_sigSlope (float):
                The slope of the sigmoid used to constrain the sample weights
                 so as to give a score simply on whether or not they were
                 included.
                Recommend keeping this to 1-5.
            fracWeighted_sigCenter (float):
                The center of the sigmoid described above.
                Recommend keeping to 0.5.
            fracWeight_penalty (float):
                The penalty applied to the sample weights loss.
            maskL1_penalty (float):
                The penalty applied to the L1 norm of the mask.
                Adjust this to reduce the probability of extracting clusters
                 with low scores.
            tol_convergence (float):
                The tolerance for the convergence of the optimization.
            window_convergence (int):
                The number of iterations to use in the convergence check.
                A regression is performed on the last window_convergence to
                 see if the best fit line has changed by less than
                 tol_convergence.
            freqCheck_convergence (int):
                The number of iterations to wait betwee checking for
                 convergence.
            verbose (bool):
                Whether to print progress.
        """

        self._n_samples = c.shape[0]
        self._n_clusters = h.shape[1]

        self._DEVICE = device

        c_tmp = helpers.scipy_sparse_to_torch_coo(c).coalesce().type(torch.float32)
        self.c = ts.SparseTensor(
            row=c_tmp.indices()[0], 
            col=c_tmp.indices()[1], 
            value=c_tmp.values(),
            sparse_sizes=c_tmp.shape,
        ).to(self._DEVICE)

        self.h = helpers.scipy_sparse_to_torch_coo(h).coalesce().type(torch.float32).to(self._DEVICE)

        w_tmp = scipy.sparse.lil_matrix((self._n_clusters, self._n_clusters))
        w_tmp[range(self._n_clusters), range(self._n_clusters)] = (w / w.max()) if w is not None else 1
        self.w = helpers.scipy_sparse_to_torch_coo(w_tmp).coalesce().type(torch.float32).to(self._DEVICE)
        # self.w = (torch.eye(len(w)) * (w / w.max())[None,:]).to_sparse().type(torch.float32).to(self._DEVICE) if w is not None else torch.eye(self._n_samples).type(torch.float32).to_sparse().to(self._DEVICE)

        self.m = m_init.to(self._DEVICE) if m_init is not None else (torch.ones(self._n_clusters)*0.1 + torch.rand(self._n_clusters)*0.05).type(torch.float32).to(self._DEVICE)
        self.m.requires_grad=True

        self._optimizer = optimizer_partial(params=[self.m]) if optimizer_partial is not None else torch.optim.Adam(params=[self.m], lr=1e-2, betas=(0.9, 0.900))
        self._scheduler = scheduler_partial(optimizer=self._optimizer) if scheduler_partial is not None else torch.optim.lr_scheduler.LambdaLR(optimizer=self._optimizer, lr_lambda=lambda x : x, last_epoch=-1, verbose=True)
        
        self._dmCEL_penalty = dmCEL_penalty
        self._sampleWeight_penalty = sampleWeight_penalty
        self._fracWeight_penalty = fracWeight_penalty
        self._maskL1_penalty = maskL1_penalty

        self._dmCEL = self._DoubleMasked_CEL(
            c=self.c,
            n_clusters=self._n_clusters,
            device=self._DEVICE,
            temp=dmCEL_temp,
            sig_slope=dmCEL_sigSlope,
            sig_center=dmCEL_sigCenter,
        )

        self._loss_fracWeighted = self._Loss_fracWeighted(
            h=self.h,
            w=self.w,
            goal_frac=fracWeighted_goalFrac,
            sig_slope=fracWeighted_sigSlope,
            sig_center=fracWeighted_sigCenter,
        )

        self._loss_sampleWeight = self._Loss_sampleWeight(
            softplus_kwargs=sampleWeight_softplusKwargs,
        )

        self._convergence_checker = self._Convergence_checker(
            tol_convergence=tol_convergence,
            window_convergence=window_convergence,
        )

        self._tol_convergence = tol_convergence
        self._window_convergence = window_convergence
        self._freqCheck_convergence = freqCheck_convergence
        self._verbose = verbose

        self._i_iter = 0
        self.losses_logger = {
            'loss': [],
            'L_cs': [],
            'L_fracWeighted': [],
            'L_sampleWeight': [],
            'L_maskL1': [],
        }

        # gc.collect()
        # torch.cuda.empty_cache()

    def fit(
        self, 
        min_iter=1e3,
        max_iter=1e4,
        verbose=True, 
        verbose_interval=100
    ):
        loss_smooth = np.nan
        diff_window_convergence = np.nan
        while self._i_iter <= max_iter:
            self._optimizer.zero_grad()

            L_cs = self._dmCEL(c=self.c, m=self.m) * self._dmCEL_penalty  ## 'cluster similarity loss'
            # L_sampleWeight = self._loss_sampleWeight(self.h, self.activate_m()) * self._sampleWeight_penalty
            # # L_sampleWeight = self._loss_sampleWeight(self.h, self.m) * self._sampleWeight_penalty
            # L_fracWeighted = self._loss_fracWeighted(self.activate_m()) * self._fracWeight_penalty
            # L_maskL1 = torch.sum(torch.abs(self.activate_m())) * self._maskL1_penalty

            # self._loss = L_cs + L_fracWeighted + L_sampleWeight + L_maskL1
            # # self._loss = L_fracWeighted + L_sampleWeight + L_maskL1
            # # self._loss = L_cs 

            # if torch.isnan(self._loss):
            #     print(f'STOPPING EARLY: loss is NaN. iter: {self._i_iter}  loss: {self._loss.item():.4f}  L_cs: {L_cs.item():.4f}  L_fracWeighted: {L_fracWeighted.item():.4f}  L_sampleWeight: {L_sampleWeight.item():.4f}  L_maskL1: {L_maskL1.item():.4f}')
            #     break

            # self._loss.backward()
            # self._optimizer.step()
            # self._scheduler.step()

            # self.m.data = torch.maximum(self.m.data , torch.as_tensor(-14, device=self._DEVICE))

            # self.losses_logger['loss'].append(self._loss.item())
            # self.losses_logger['L_cs'].append(L_cs.item())
            # self.losses_logger['L_fracWeighted'].append(L_fracWeighted.item())
            # self.losses_logger['L_sampleWeight'].append(L_sampleWeight.item())
            # self.losses_logger['L_maskL1'].append(L_maskL1.item())

            # if self._i_iter%self._freqCheck_convergence==0 and self._i_iter>self._window_convergence and self._i_iter>min_iter:
            #     diff_window_convergence, loss_smooth, converged = self._convergence_checker(self.losses_logger['loss'])
            #     if converged:
            #         print(f"STOPPING: Convergence reached in {self._i_iter} iterations.  loss: {self.losses_logger['loss'][-1]:.4f}  loss_smooth: {loss_smooth:.4f}")
            #         break

            # if verbose and self._i_iter % verbose_interval == 0:
            #     print(f'iter: {self._i_iter}:  loss_total: {self._loss.item():.4f}  lr: {self._scheduler.get_last_lr()[0]:.5f}   loss_cs: {L_cs.item():.4f}  loss_fracWeighted: {L_fracWeighted.item():.4f}  loss_sampleWeight: {L_sampleWeight.item():.4f}  loss_maskL1: {L_maskL1.item():.4f}  diff_loss: {diff_window_convergence:.4f}  loss_smooth: {loss_smooth:.4f}')
            #     # print(torch.isnan(self.m).sum())
            # self._i_iter += 1


    def predict(
        self,
        m_threshold=0.5
    ):   
        h_ts = helpers.torch_to_torchSparse(self.h)

        self.m_bool = (self.activate_m() > m_threshold).squeeze().cpu()
        h_ts_bool = h_ts[:, self.m_bool]
        n_multiples = (h_ts_bool.sum(1) > 1).sum()
        print(f'WARNING: {n_multiples} samples are matched with multiple clusters. Consider increasing the sample_weight_penalty during training.') if n_multiples > 0 else None
        h_preds = (h_ts_bool * (torch.arange(self.m_bool.sum(), device=self._DEVICE)[None,:]+1)).detach().cpu()
        # h_preds[h_preds==0] = -1

        if h_preds.numel() == 0:
            print(f'WARNING: No predictions made.  m_threshold: {m_threshold}')
            return None, None

        # preds = torch.max(h_preds, dim=1)[0]
        preds = h_preds.max(dim=1) - 1
        preds[torch.isinf(preds)] = -1
        
        h_m = (h_ts * self.activate_m()[None,:]).detach().cpu().to_dense()
        confidence = h_m.var(1) / h_m.mean(1)

        self.scores_clusters = helpers.diag_sparse(self.w).cpu()[self.m_bool]
        self.scores_samples = (h_preds * self.scores_clusters[None,:]).max(1)
        
        self.preds = preds
        self.confidence = confidence
        
        return self.preds, self.confidence, self.scores_samples, self.m_bool

    def activate_m(self):
        return self._dmCEL.activation(self.m)


    class _DoubleMasked_CEL:
        def __init__(
            self,
            c,
            n_clusters,
            device='cpu',
            temp=1,
            sig_slope=5,
            sig_center=0.5,
        ):
            # self.labels = torch.arange(n_clusters, device=device, dtype=torch.int64)
            # self.CEL = torch.nn.CrossEntropyLoss(reduction='none')
            # self.CEL = lambda x : helpers.diag_sparse(torch.sparse.log_softmax(x, dim=1).coalesce())
            # self.CEL = lambda x : -helpers.diag_sparse(torch.sparse.log_softmax(x/self.temp, dim=1).coalesce())
            # self.CEL = lambda x : -helpers.diag_sparse(helpers.ts_logSoftmax(x, temperature=self.temp).coalesce())
            self.CEL = lambda x :  - helpers.ts_logSoftmax(x, temperature=self.temp, shift=None).get_diag()
            self.temp = temp
            self.activation = self.make_sigmoid_function(sig_slope, sig_center)
            self.device=device
            
            # self.w = w
            
            # self.idx_diag = torch.arange(n_clusters, device=device, dtype=torch.int64)

            # self.m_eye = torch.sparse_coo_tensor(
            #     indices=torch.vstack((torch.arange(n_clusters), torch.arange(n_clusters))).to(device),
            #     values=torch.ones(n_clusters).to(device),
            #     size=(n_clusters, n_clusters)
            # ).coalesce()
            # self.m_eye = torch.eye(n_clusters, device=device).to(device)

            self.worst_case_loss = self.CEL(
                # (torch.logical_not(torch.eye(n_clusters, device=device)) + torch.eye(n_clusters, device=device)*c.diag()[None,:]).type(torch.float32) / self.temp,
                # c / self.temp,
                # self.labels
                c,
            )
            self.best_case_loss = self.CEL(
                # torch.sparse_coo_tensor(
                #     indices=torch.vstack((torch.arange(n_clusters), torch.arange(n_clusters))).to(device),
                #     # values=torch_helpers.diag_sparse(c), 
                #     values=c.get_diag().type(torch.float32), 
                #     size=(n_clusters, n_clusters)
                # )
                ts.tensor.SparseTensor(
                    row=torch.arange(n_clusters, device=device, dtype=torch.int64),
                    col=torch.arange(n_clusters, device=device, dtype=torch.int64),
                    value=c.get_diag().type(torch.float32),
                    sparse_sizes=(n_clusters, n_clusters),
                    # indices=torch.vstack((torch.arange(n_clusters), torch.arange(n_clusters))).to(device),
                    # # values=torch_helpers.diag_sparse(c), 
                    # values=c.get_diag().type(torch.float32), 
                    # size=(n_clusters, n_clusters)
                )
            )
            self.worst_minus_best = self.worst_case_loss - self.best_case_loss  + 1e-8

        def make_sigmoid_function(
            self,
            sig_slope=5,
            sig_center=0.5,
        ):
            return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))
            
        def __call__(self, c, m):
            mp = self.activation(m)  ## constrain to be 0-1

            ###### cm = c * mp[None,:]  ## 'c masked'. Mask only applied to columns.
            # cm = c @ torch.sparse_coo_tensor(
            #     indices=torch.vstack((torch.arange(c.shape[0]), torch.arange(c.shape[0]))).to(self.device),
            #     values=mp,
            #     size=(c.shape[0], c.shape[0])
            # )  ## 'c masked'. Mask only applied to columns.
            # cm = c @ (self.m_eye*mp[None,:])  ## 'c masked'. Mask only applied to columns.
            # cm = c * m[None,:]
            # cm = c
            ###### cm[self.idx_diag, self.idx_diag] = c.diagonal()
            # cm = c * mp.sum()

            # cm = c * mp[None,:]
            # # cm = cm.set_diag(c.get_diag())
            # cm = helpers.ts_setDiag_lowMem(cm, c.get_diag())
            # lv = self.CEL(cm)

            lv = self.CEL(helpers.ts_setDiag_lowMem(c * mp[None,:], c.get_diag()))

            # lv = self.CEL(
            #     cm/self.temp, 
            #     # self.labels
            #     )  ## 'loss vector' showing loss of each row (each cluster)
            # print(self.worst_minus_best)

            lv_norm = (lv - self.best_case_loss) / self.worst_minus_best

            # print(lv_norm @ mp)
            # self.test = mp
            
            l = (lv_norm @ mp) / mp.sum()  ## 'loss'
            
            # l = ((lv/self.w) @ mp) / mp.sum()  ## 'loss'

            return l
            # return 1

    class _Loss_fracWeighted:
        def __init__(
            self,
            h,
            w,
            goal_frac=1.0,
            sig_slope=5,
            sig_center=0.5,
        ):
            # self.h_w = h.type(torch.float32) * w[None,:]
            self.h_w = h.type(torch.float32) @ w
            # self.h_w = h.type(torch.float32)
        
            self.goal_frac = goal_frac
            self.sigmoid = self.make_sigmoid_function(sig_slope, sig_center)
            
        def make_sigmoid_function(
            self,
            sig_slope=5,
            sig_center=0.5,
        ):
            return lambda x : 1 / (1 + torch.exp(-sig_slope*(x-sig_center)))

        def generate_sampleWeights(self, m):
            return self.h_w @ m
        def activate_sampleWeights(self, sampleWeights):
            return self.sigmoid(sampleWeights)

        def __call__(self, m):
            return ((self.activate_sampleWeights(self.generate_sampleWeights(m))).mean() - self.goal_frac)**2

    class _Loss_sampleWeight:
        def __init__(
            self,
            softplus_kwargs=None,
        ):
            self.softplus = torch.nn.Softplus(**softplus_kwargs)
            self.weight_high = 1
            self.offset = self.activate_sampleWeights(torch.ones(1, dtype=torch.float32)).max()
        
        def generate_sampleWeights(self, h, m):
            return h @ m
        def activate_sampleWeights(self, sampleWeights):
            return self.softplus(sampleWeights - 1 - 4/self.softplus.beta)*self.weight_high

        def __call__(self, h, m):
            return (self.activate_sampleWeights(self.generate_sampleWeights(h, m))).mean()

    class _Convergence_checker:
        def __init__(
            self,
            tol_convergence=1e-2,
            window_convergence=100,
        ):
            self.window_convergence = window_convergence
            self.tol_convergence = tol_convergence

            self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

        def OLS(self, y):
            X = self.line_regressor
            theta = torch.inverse(X.T @ X) @ X.T @ y
            y_rec = X @ theta
            bias = theta[-1]
            theta = theta[:-1]

            return theta, y_rec, bias

        def __call__(
            self,
            loss_history,
        ):
            loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
            theta, y_rec, bias = self.OLS(y=loss_window)

            diff_window_convergence = (y_rec[-1] - y_rec[0])
            loss_smooth = loss_window.mean()
            converged = True if torch.abs(diff_window_convergence) < self.tol_convergence else False
            return diff_window_convergence.item(), loss_smooth.item(), converged
            

    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses_logger['loss'], linewidth=4)
        plt.plot(self.losses_logger['L_cs'])
        plt.plot(self.losses_logger['L_fracWeighted'])
        plt.plot(self.losses_logger['L_sampleWeight'])
        plt.plot(self.losses_logger['L_maskL1'])

        plt.legend(self.losses_logger.keys())
        plt.xlabel('iteration')
        plt.ylabel('loss')

    def plot_clusterWeights(self, plot_raw_m=False):
        plt.figure()
        if plot_raw_m:
            plt.hist(self.m.detach().cpu().numpy(), bins=50)
        plt.hist(self.activate_m().detach().cpu().numpy(), bins=50)
        plt.xlabel('cluster weight')
        plt.ylabel('count')

    def plot_clusterScores(self, bins=100):
        if hasattr(self, 'm_bool'):
            m_bool = self.m_bool
            confidence = self.confidence
            preds = self.preds
        else:
            preds, confidence, scores_samples, m_bool = self.predict()
            if preds is None:
                print('Plot failed: preds is None.')
                return None
        scores = helpers.diag_sparse(self.w).cpu()[m_bool.cpu()]

        plt.figure()
        plt.hist(scores, bins=bins, log=True)
        plt.xlabel('cluster score')
        plt.ylabel('count')

        plt.figure()
        plt.hist(confidence[preds >= 0], bins=bins, log=True)
        plt.xlabel('confidence')
        plt.ylabel('count')

    def plot_sampleWeights(self):
        sampleWeights = self._loss_sampleWeight.generate_sampleWeights(self.h, self.activate_m()).detach().cpu().numpy()

        plt.figure()
        plt.hist(sampleWeights, bins=50)
        plt.xlabel('sample weight')
        plt.ylabel('count')

    def plot_labelCounts(self):
        if hasattr(self, 'preds'):
            preds = self.preds
        else:
            preds, confidence, scores_samples, m_bool = self.predict()
            if preds is None:
                print('Skipping plot_labelCounts: preds is None.')
                return None
            
        labels_unique, label_counts = np.unique(preds, return_counts=True)

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].bar(labels_unique, label_counts)
        axs[0].set_xlabel('label')
        axs[0].set_ylabel('count')
        axs[1].hist(label_counts[labels_unique>=0], bins=25)
        axs[1].set_xlabel('n_roi per cluster')
        axs[1].set_ylabel('counts')

        return fig, axs

    def plot_c_threshold_matrix(self, m_threshold=0.5):
        mt = self.activate_m() > m_threshold
        plt.figure()
        plt.imshow(self.c[mt, mt].detach().cpu(), aspect='auto')

    def plot_c_masked_matrix(self, m_threshold=0.5, **kwargs_imshow):
        import sparse
        mt = self.activate_m() > m_threshold
        # return (self.c * mt[None,:])
        cdm = (sparse.COO(self.c.to_scipy()) * mt[None,:].cpu().numpy()  * mt[:, None].cpu().numpy()).tocsr().tolil()
        cdm[list(self._dmCEL.idx_diag.cpu().numpy()), list(self._dmCEL.idx_diag.cpu().numpy())] = self.c.get_diag().cpu()
        plt.figure()
        plt.imshow(cdm.toarray(), aspect='auto')