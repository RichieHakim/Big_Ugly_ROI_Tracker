# # widen jupyter notebook window
# from IPython.display import display, HTML
# display(HTML("<style>.container {width:95% !important; }</style>"))

# check environment
import os
# print(f'Conda Environment: ' + os.environ['CONDA_DEFAULT_ENV'])

from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import scipy.stats
import time

class ROInet:
    def __init__(
        self,
            # dir_github='/media/rich/Home_Linux_partition/github_repos/',
            # 'dir_s2p': '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/suite2p_o2_output/jobNum_0/suite2p/plane0/',
            # 'path_params_nnTraining': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/params.json',
            # 'path_state_dict': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/ConvNext_tiny__1_0_unfrozen__simCLR.pth',
            # 'path_classifier_vars': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/classifier.pkl',
            # 'pref_saveFigs': False,
        device='cpu',
            # 'classes_toInclude': [0,1,2]
    ):
        # import sys
        # path_script, path_params, dir_save = sys.argv
        # dir_save = Path(dir_save)
                        
        # import json
        # with open(path_params, 'r') as f:
        #     params = json.load(f)

        # import shutil
        # shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

        # params = {
        #     'dir_github': '/media/rich/Home_Linux_partition/github_repos/',
        #     'dir_s2p': '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/suite2p_o2_output/jobNum_0/suite2p/plane0/',
        #     'path_params_nnTraining': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/params.json',
        #     'path_state_dict': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/ConvNext_tiny__1_0_unfrozen__simCLR.pth',
        #     'path_classifier_vars': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/classifier.pkl',
        #     'pref_saveFigs': False,
        #     'useGPU': True,
        #     'classes_toInclude': [0,1,2]
        # }

        # import sys
        # sys.path.append(params['dir_github'])

        # print('starting: importing custom modules')
        # from basic_neural_processing_modules import pickle_helpers, indexing, torch_helpers

        self.device = device

        from NBAP.pipeline_2pRAM_faceRhythm.classify_ROIs import util
        print('completed: importing custom modules')


        dir_save_network_files = str(Path(dir_save).resolve() / 'network_files')

        print(f"starting: downloading network from {params['gdriveID_networkFiles']}")
        import gdown
        gdown.download_folder(id=params['gdriveID_networkFiles'], output=dir_save_network_files, quiet=True, use_cookies=False)
        sys.path.append(dir_save_network_files)
        import model
        print(f"completed: downloading network")

        path_state_dict = str(Path(dir_save_network_files).resolve() / params['fileName_state_dict'])
        path_nnTraining = str(Path(dir_save_network_files).resolve() / params['fileName_params_nnTraining'])
        # path_model = str(Path(dir_save_network_files).resolve() / params['fileName_model'])
        path_classifier = str(Path(dir_save_network_files).resolve() / params['fileName_classifier'])




        path_stat = str(Path(params['dir_s2p']) / 'stat.npy')
        path_ops = str(Path(params['dir_s2p']) / 'ops.npy')

        print('starting: loading stat file')
        sf_all = util.import_multiple_stat_files(   
            paths_statFiles=[path_stat],
            out_height_width=[36,36],
            max_footprint_width=1441,
            plot_pref=True
        )
        print('completed: loading stat file')

        print('starting: resizing ROIs')
        sf_ptiles = np.array([np.percentile(np.sum(sf>0, axis=(1,2)), 90) for sf in tqdm(sf_all)])
        scales_forRS = (250/sf_ptiles)**0.6
        sf_rs = [np.stack([util.resize_affine(img, scale=scales_forRS[ii], clamp_range=True) for img in sf], axis=0) for ii, sf in enumerate(tqdm(sf_all))]

        sf_all_cat = np.concatenate(sf_all, axis=0)
        sf_rs_concat = np.concatenate(sf_rs, axis=0)
        print('completed: resizing ROIs')

        import scipy.signal

        figs, axs = plt.subplots(2,1, figsize=(7,10))
        axs[0].plot(np.sum(sf_all_cat > 0, axis=(1,2)))
        axs[0].plot(scipy.signal.savgol_filter(np.sum(sf_all_cat > 0, axis=(1,2)), 501, 3))
        axs[0].set_xlabel('ROI number');
        axs[0].set_ylabel('mean npix');
        axs[0].set_title('ROI sizes raw')

        axs[1].plot(np.sum(sf_rs_concat > 0, axis=(1,2)))
        axs[1].plot(scipy.signal.savgol_filter(np.sum(sf_rs_concat > 0, axis=(1,2)), 501, 3))
        axs[1].set_xlabel('ROI number');
        axs[1].set_ylabel('mean npix');
        axs[1].set_title('ROI sizes resized')

        if params['pref_saveFigs']:
            plt.savefig(str(Path(dir_save) / 'ROI_sizes.png'))

        print('starting: making dataloader')
        transforms_classifier = torch.nn.Sequential(
            util.ScaleDynamicRange(scaler_bounds=(0,1)),
            
            torchvision.transforms.Resize(
                size=(224, 224),
        #         size=(180, 180),
        #         size=(72, 72),        
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR), 
            
            util.TileChannels(dim=0, n_channels=3),
        )

        scripted_transforms_classifier = torch.jit.script(transforms_classifier)


        dataset_labeled = util.dataset_simCLR(
                X=torch.as_tensor(sf_rs_concat, device='cpu', dtype=torch.float32),
                y=torch.as_tensor(torch.zeros(sf_rs_concat.shape[0]), device='cpu', dtype=torch.float32),
                n_transforms=1,
                class_weights=np.array([1]),
                transform=scripted_transforms_classifier,
                DEVICE='cpu',
                dtype_X=torch.float32,
            )
            
        dataloader_labeled = out = torch.utils.data.DataLoader( 
                dataset_labeled,
                batch_size=8,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=10,
                persistent_workers=True,
        #         prefetch_factor=2
        )
        print('completed: making dataloader')

        print('starting: importing network parameters and making network')
        import json
        with open(path_nnTraining) as f:
            params_nnTraining = json.load(f)

        model_nn = model.make_model(
            torchvision_model=params_nnTraining['torchvision_model'],
            n_block_toInclude=params_nnTraining['n_block_toInclude'],
            pre_head_fc_sizes=params_nnTraining['pre_head_fc_sizes'],
            post_head_fc_sizes=params_nnTraining['post_head_fc_sizes'],
            head_nonlinearity=params_nnTraining['head_nonlinearity'],
            image_shape=[3, 224, 224],
        #     image_shape=[params_nnTraining['augmentation']['TileChannels']['n_channels']] + params_nnTraining['augmentation']['WarpPoints']['img_size_out']
        );

        for param in model_nn.parameters():
            param.requires_grad = False
        model_nn.eval();

        # model_nn.load_state_dict(torch.load(params['path_state_dict']))
        model_nn.load_state_dict(torch.load(path_state_dict))

        model_nn = model_nn.to(self.device)
        print('completed: importing network parameters and making network')

        print(f'starting: running data through network {time.ctime()}')
        features_nn = torch.cat([model_nn(data[0][0].to(self.device)).detach() for data in tqdm(dataloader_labeled)], dim=0).cpu()
        print(f'completed: running data through network {time.ctime()}')


    def import_multiple_stat_files(
        self,
        paths_statFiles=None, 
        dir_statFiles=None, 
        fileNames_statFiles=None,
        out_height_width=[36,36], 
        max_footprint_width=241, 
        plot_pref=True
    ):
        """
        Imports multiple stat files.
        RH 2021 
        
        Args:
            paths_statFiles (list):
                List of paths to stat files.
                Elements can be either str or pathlib.Path.
            dir_statFiles (str):
                Directory of stat files.
                Optional: if paths_statFiles is provided, this
                argument is ignored.
            fileNames_statFiles (list):
                List of file names of stat files.
                Optional: if paths_statFiles is provided, this
                argument is ignored.
            out_height_width (list):
                [height, width] of the output spatial footprints.
            max_footprint_width (int):
                Maximum width of the spatial footprints.
            plot_pref (bool):
                If True, plots the spatial footprints.

        Returns:
            stat_all (list):
                List of stat files.
        """
        if paths_statFiles is None:
            paths_statFiles = [Path(dir_statFiles) / fileName for fileName in fileNames_statFiles]

        sf_all_list = [self.statFile_to_centered_spatialFootprints(path_statFile=path_statFile,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref)
                    for path_statFile in paths_statFiles]
        return sf_all_list

    def statFile_to_centered_spatialFootprints(
        self,
        path_statFile=None, 
        statFile=None, 
        out_height_width=[36,36], 
        max_footprint_width=241, 
        plot_pref=True
    ):
        """
        Converts a stat file to a list of spatial footprint images.
        RH 2021

        Args:
            path_statFile (pathlib.Path or str):
                Path to the stat file.
                Optional: if statFile is provided, this
                argument is ignored.
            statFile (dict):
                Suite2p stat file dictionary
                Optional: if path_statFile is provided, this
                argument is ignored.
            out_height_width (list):
                [height, width] of the output spatial footprints.
            max_footprint_width (int):
                Maximum width of the spatial footprints.
            plot_pref (bool):
                If True, plots the spatial footprints.
        
        Returns:
            sf_all (list):
                List of spatial footprints images
        """
        assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "RH: 'out_height_width' must be list of 2 EVEN integers"
        assert max_footprint_width%2 != 0 , "RH: 'max_footprint_width' must be odd"
        if statFile is None:
            stat = np.load(path_statFile, allow_pickle=True)
        else:
            stat = statFile
        n_roi = stat.shape[0]
        
        # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
        sf_big_width = max_footprint_width # make odd number
        sf_big_mid = sf_big_width // 2

        sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
        for ii in range(n_roi):
            sf_big[ii , stat[ii]['ypix'] - np.int16(stat[ii]['med'][0]) + sf_big_mid, stat[ii]['xpix'] - np.int16(stat[ii]['med'][1]) + sf_big_mid] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

        sf = sf_big[:,  
                    sf_big_mid - out_height_width[0]//2:sf_big_mid + out_height_width[0]//2,
                    sf_big_mid - out_height_width[1]//2:sf_big_mid + out_height_width[1]//2]
        if plot_pref:
            plt.figure()
            plt.imshow(np.max(sf, axis=0)**0.2)
            plt.title('spatial footprints cropped MIP^0.2')
        
        return sf