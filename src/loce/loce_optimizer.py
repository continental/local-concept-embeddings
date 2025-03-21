'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import torch
from torch import Tensor
from torch.optim import AdamW
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F

from data_structures import MSCOCOAnnotationsProcessor
from .loce import LoCE, LoCEMultilayerStorage, LoCEMultilayerStorageSaver, LoCEMultilayerStorageStats
from .loce_utils import get_projection, ImageLoader, LoCEActivationsTensorExtractor
from .loce_semantic_segmenter import AbstractSemanticSegmenter, MSCOCOSemanticSegmentationLoader, MSCOCORectangleSegmenter, MSCOCOEllipseSegmenter
from xai_utils.files import read_json, mkdir

import os
from typing import Dict, Any, Iterable, Tuple, List, Literal


EPSILON = 0.0001

class TorchCustomLoCEBatchOptimizer:

    def __init__(self,
                 loce_init: Literal["zeros", "ones", "random_uniform", "random_normal"] = "zeros",
                 seed: int = None,
                 objective_type: Literal["bce", "mse", "mae"] = "bce",
                 denoise_activations: bool = False
                 ) -> None:
        """
        Args:
            loce_init (str: Literal["zeros", "ones", "random_uniform", "random_normal"]): initialize loce vector with torch.zeros(), torch.ones(), torch.rand() or torch.randn()
            seed (int = None): seed for "random_uniform", "random_normal" loce_init
            objective_type (str): optimization objective
            denoise_activations (bool): denoising flag
        """
        self.seed = seed
        self.loce_init = loce_init
        self.objective_type = objective_type
        self.denoise_activations = denoise_activations

    # modified BCE from Net2Vec paper
    @staticmethod
    def _objective_bce(projection_vector: Tensor,
                       target: Tensor,
                       acts_tensor: Tensor,
                       alphas_batch: float):        

        weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

        preds = torch.sigmoid(weighted_activations)

        batch_size = preds.size(0)

        alpha = alphas_batch.view(batch_size, 1, 1, 1)
        beta = 1 - alpha

        # loss per sample
        pseudo_bce = -1./ batch_size * torch.sum(alpha*torch.mul(target, preds) + beta*torch.mul(1-target, 1-preds), dim=[1,2,3])
            
        return pseudo_bce
    
    @staticmethod
    def _objective_mae_reg(projection_vector: Tensor,
                          mask_tensor_binary: Tensor,
                          acts_tensor: Tensor,
                          alpha: float = None):

            weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

            preds = torch.sigmoid(weighted_activations)

            mae = (torch.abs(mask_tensor_binary - preds)).mean()

            l1 = torch.norm(projection_vector, 1) / len(projection_vector)
            l2 = torch.norm(projection_vector, 2) / len(projection_vector)

            regularization = (l1 + l2)

            return mae + regularization
    
    @staticmethod
    def _objective_mse_reg(projection_vector: Tensor,
                          mask_tensor_binary: Tensor,
                          acts_tensor: Tensor,
                          alpha: float = None):

            weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

            preds = torch.sigmoid(weighted_activations)

            mae = ((mask_tensor_binary - preds) ** 2).mean()

            l1 = torch.norm(projection_vector, 1) / len(projection_vector)
            l2 = torch.norm(projection_vector, 2) / len(projection_vector)

            regularization = (l1 + l2)

            return mae + regularization

    def _get_loces(self,
                   baseline_masks: Tensor,
                   acts: Tensor,
                   batch_size: int = 64,
                   lr: float = 0.1,
                   epochs: int = 50
                   ) -> Dict[str, Any]:
        """
        Optimize LoCEs

        Args:
            baseline_masks (Tensor): baseline segmentation
            acts (Tensor): activations of sample
        
        Kwargs:
            batch_size (int): batch size 
            lr (float): learning rate of optimizer
            epochs (int): optimization epochs
        
        Returns:
            (Dict[str, Any]) of optimization results
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        b, c, w, h = acts.shape

        acts_device = acts
        seg_masks_device = baseline_masks.float()

        # for balancing background and foreground pixels (per sample)
        # alpha = 1 - (num_fg_pixels / num_pixels) 
        alphas = 1 - (seg_masks_device > 0).float().mean(dim=(1, 2))

        # select loce optimization objective
        if self.objective_type == "mae":
            criterion = self._objective_mae_reg
        elif self.objective_type == "mse":
            criterion = self._objective_mse_reg
        else:
            criterion = self._objective_bce

        # loce init function
        if self.loce_init == "ones":
            loce_init_fn = torch.ones
        elif self.loce_init == "random_uniform":
            loce_init_fn = torch.rand
            if not self.seed is None:
                torch.manual_seed(self.seed)
        elif self.loce_init == "random_normal":
            loce_init_fn = torch.randn
            if not self.seed is None:
                torch.manual_seed(self.seed)
        else:
            loce_init_fn = torch.zeros

        loces = []

        losses = []

        idxs = torch.arange(len(acts_device))       

        start_idx = 0
        while start_idx < len(acts_device):
            selected_idxs = idxs[start_idx:start_idx+batch_size]
            start_idx = start_idx + batch_size

            acts_batch = acts_device[selected_idxs].to(device)
            mask_batch = seg_masks_device[selected_idxs].unsqueeze(1).to(device)

            alphas_batch = alphas[selected_idxs].to(device)
            
            loces_batch = loce_init_fn((len(selected_idxs), c, 1, 1), device=device)
            loces_batch.requires_grad_()

            opt = AdamW([loces_batch], lr=lr)

            batch_losses = []
            
            for i in range(epochs):                

                per_sample_loss = criterion(loces_batch, mask_batch, acts_batch, alphas_batch)

                batch_losses.append(per_sample_loss.detach().cpu().numpy())

                opt.zero_grad()
                per_sample_loss.mean().backward()
                opt.step()

            losses.append(np.array(batch_losses))

            loces.append(loces_batch.detach().squeeze())
            #with torch.no_grad():
                    #loce.clamp_(None, None)

        #print(f"\tLoss ({init_fn}):", result['fun'])
        return {'x': torch.vstack(loces).cpu().numpy(), 'fun': np.hstack(losses).T}

    def optimize_loces(self,
                       segmentations: Tensor,
                       activations: Dict[str, Tensor],
                       batch_size: int = 64,
                       lr: float = 0.1,
                       epochs: int = 50
                       ) -> Dict[str, np.ndarray]:
        """
        Get prototypes of LoCEs for a single sample in all given layers

        Args:
            segmentations (Tensor[B,W,H]): segmentation masks (reshaped)
            activations (Dict[str, Tensor[B,C,W,H]]): per-layer dictionary of activations (reshaped)

        Kwargs:
            batch_size (int): batch size 
            lr (float): learning rate of optimizer
            epochs (int): optimization epochs
        Returns:
            (Dict[str, np.ndarray]) per-layer LoCEs
        """

        loces = {l: None for l in activations.keys()}

        for layer, acts in activations.items():

            #if self.denoise_activations:
            #    cutoffs = compute_quantile_cutoffs(acts_current)
            #    acts_current = threshold_activations(acts_current, cutoffs)

            result_ones = self._get_loces(segmentations, acts, batch_size, lr, epochs)

            # JUST DONT REPEAT THAT FOR THE SAKE OF OUR LORD JESUS CHRIST
            concept_vectors = result_ones['x']

            loces[layer] = concept_vectors

        return loces


class LoCEOptimizationEngineMSCOCO:

    coco_segmenters = {
        'original': MSCOCOSemanticSegmentationLoader,
        'rectangle': MSCOCORectangleSegmenter,
        'ellipse': MSCOCOEllipseSegmenter
        }

    mscoco_tags = {
        # not used in experiments - commented out
        1: 'person',

        # vehicles
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',

        # animals
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe'
        }

    def __init__(self,
                 batch_optimizer: TorchCustomLoCEBatchOptimizer,
                 activations_extractor: LoCEActivationsTensorExtractor,
                 mscoco_imgs_path: str = "./data/mscoco2017val/val2017/",
                 mscoco_default_annots: str = "./data/mscoco2017val/annotations/instances_val2017.json",
                 mscoco_processed_annots: str = "./data/mscoco2017val/processed/",
                 out_base_dir: str = "./experiment_outputs/optimized_loces_coco/",
                 n_imgs_per_category: int = 500,
                 target_shape: Tuple[int, int] = (100, 100),
                 mscoco_tags: Dict[int, str] = None,
                 image_loader: ImageLoader = ImageLoader(),
                 ) -> None:
        """
        Args:
            batch_optimizer (TorchCustomLoCEBatchOptimizer): batch optimizer
            activations_extractor (LoCEActivationsTensorExtractor): activations extractor

        Kwargs:
            out_base_dir (str = "./experiment_outputs/optimized_loces/"): base directory for outputs, subdirs will be created
            processor (BaseImageProcessor = None): Processor from Hugging Face
        """
        self.batch_optimizer = batch_optimizer
        self.activations_extractor = activations_extractor
        self.propagator_tag = activations_extractor.propagator_tag.lower()
        self.mscoco_imgs_path = mscoco_imgs_path
        self.mscoco_default_annots = mscoco_default_annots
        self.mscoco_processed_annots = mscoco_processed_annots
        self.out_base_dir = out_base_dir
        self.n_imgs_per_category = n_imgs_per_category
        self.target_shape = target_shape
        self.img_loader = image_loader
        
        if mscoco_tags is not None:
            self.mscoco_tags = mscoco_tags

        self.annots = self._load_coco_annots()

        mkdir(out_base_dir)

    def run_optimization_all_segmenters(self,
                                        batch_size: int = 64,
                                        verbose: bool = False
                                        ) -> None:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """
        for segmenter_tag in self.coco_segmenters.keys():

            self.run_optimization(segmenter_tag, batch_size, verbose)

    def run_optimization(self,
                         segmenter_tag: Literal["original", "rectangle", "ellipse"],
                         batch_size: int = 64,
                         verbose: bool = False
                         ) -> None:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """

        out_dir = os.path.join(self.out_base_dir, f'loce_{segmenter_tag}_{self.propagator_tag}')
        mkdir(out_dir)

        # analyze each category
        for category_id, category in tqdm(self.mscoco_tags.items()):
            
            coco_category_annot = self.annots[category]

            # get segmenter instance
            segmenter = self.coco_segmenters[segmenter_tag](coco_category_annot, category_id)

            # get saver
            loce_saver = LoCEMultilayerStorageSaver(out_dir)

            # get images to optimize (avoids repetition if already was optimized)
            images_to_optimize = self._get_optimization_image_names(coco_category_annot, category_id, loce_saver)

            self._optimize_one_category_batchwise(category_id, images_to_optimize, segmenter, loce_saver, batch_size)

            if verbose:
                # output stats for this category
                LoCEMultilayerStorageStats(out_dir, min_seg_area=0.0, max_seg_area=1.0).stats_one_category(category_id)

    def _get_optimization_image_names(self,
                                      coco_category_annot,
                                      category_id,
                                      loce_saver
                                      ) -> List[str]:
        # get all category image names
        coco_imgs = sorted([a['file_name'] for a in coco_category_annot['images']])
        # explicitly check if img exists
        coco_imgs = [img for img in coco_imgs if os.path.exists(os.path.join(self.mscoco_imgs_path, img))]
        # limit number of images per category
        coco_imgs = coco_imgs[:self.n_imgs_per_category]
        # check if storage already exists (if image was optimized)
        images_to_optimize = self._check_existing_storages(coco_imgs, category_id, loce_saver)

        return images_to_optimize
        
    def _optimize_one_category_batchwise(self,
                                         category_id: int,
                                         images_to_optimize: List[str],
                                         segmenter_instance: AbstractSemanticSegmenter,
                                         loce_saver: LoCEMultilayerStorageSaver,
                                         batch_size: int
                                         ) -> List[LoCEMultilayerStorage]:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """

        # runnning in batch
        start_idx = 0
        while start_idx < len(images_to_optimize):
            
            # get batch of images
            batch_imgs = images_to_optimize[start_idx:start_idx+batch_size]
            start_idx = start_idx + batch_size

            seg_masks, seg_masks_reshaped, imgs_used = self._get_segmentations(batch_imgs, segmenter_instance)
            
            activations = self._get_reshaped_activations(imgs_used)

            loces = self.batch_optimizer.optimize_loces(seg_masks_reshaped, activations, batch_size=batch_size)

            #loce_storages = self._wrap_loce_storages(loces, imgs_used, seg_masks, category_id, seg_masks_reshaped, activations)
            loce_storages = self._wrap_loce_storages(loces, imgs_used, seg_masks, category_id, activations)        

            self._save_loce_storages(loce_saver, loce_storages)

    @staticmethod
    def _check_existing_storages(images_to_check: Iterable[str],
                                 category_id: int,
                                 saver: LoCEMultilayerStorageSaver,
                                 ):
        images_to_optimize = []
        existing_storages = os.listdir(saver.working_directory)

        for img in images_to_check:
            future_storage_name = os.path.basename(saver.get_loce_storage_path_for_img_name(img, category_id)[0])
            if future_storage_name not in existing_storages:
                images_to_optimize.append(img)

        return images_to_optimize


    def _save_loce_storages(self,
                            storage_saver: LoCEMultilayerStorageSaver,
                            storages: Iterable[LoCEMultilayerStorage]
                            ):

        for storage in storages:

            image_name = os.path.basename(storage.image_path)
            category_id = storage.segmentation_category_id

            out_path_pkl, out_path_err = storage_saver.get_loce_storage_path_for_img_name(image_name, category_id)

            try:
                if not os.path.exists(out_path_pkl):
                    storage_saver.save(storage, out_path_pkl)
            except:
                open(out_path_err, 'a').close()
            

    def _wrap_loce_storages(self,
                            loces: Dict[str,np.ndarray],
                            imgs_used: Iterable[str],
                            segmentation_masks: Iterable[np.ndarray],
                            segmentation_category_id: int,
                            #segmentations_reshaped: Tensor,
                            activations_reshaped: Dict[str, Tensor]
                            ) -> List[LoCEMultilayerStorage]:
        
        storages = []

        for idx, (img_name, seg_msk) in enumerate(zip(imgs_used, segmentation_masks)):
            img_path = os.path.join(self.mscoco_imgs_path, img_name)
            storage = LoCEMultilayerStorage(img_path, None, seg_msk, segmentation_category_id)

            for layer in loces.keys():

                loce_current = loces[layer][idx]
                acts_current = activations_reshaped[layer][idx].detach().cpu().numpy()
                seg_current = segmentation_masks[idx]
                #seg_current = segmentations_reshaped[idx].detach().cpu().numpy()
                loce_proj = get_projection(loce_current, acts_current)
                bin_proj = resize(loce_proj, seg_current.shape) > 0.5
                #loce_loss = np.abs(seg_current - loce_proj / 255.).sum() / seg_current.size
                bin_seg = seg_current.astype(bool)
                loce_loss_iou = metrics.jaccard_score(bin_seg.flatten(), bin_proj.flatten())

                storage.set_loce(layer, LoCE(loce_current, loce_loss_iou, loce_proj))

            storages.append(storage)

        return storages

    def _load_coco_annots(self) -> Dict[str, Any]:
        """
        Load and process annotations of MSCOCO with MSCOCOAnnotationsProcessor

        Returns:
            annots (Dict[str, Any]): dictionary with MSCOCO annotations
        """
        annots = {}

        for v_id, v_name in self.mscoco_tags.items():

            annot_path = f"{self.mscoco_processed_annots}{v_name}_annotations_2017.json"

            try:            
                coco_annot = read_json(annot_path)
            except:
                mcp = MSCOCOAnnotationsProcessor(self.mscoco_imgs_path, self.mscoco_default_annots, self.mscoco_processed_annots)
                mcp.select_relevant_annotations_by_categtory(v_id, f"{v_name}_annotations_2017.json")

                coco_annot = read_json(annot_path)

            annots[v_name] = coco_annot

        return annots
    
    def _get_segmentations(self,
                           img_names: Iterable[str],
                           segmenter: AbstractSemanticSegmenter,
                           min_seg_area: float = 0.0,
                           max_seg_area: float = 1.0):

        seg_masks = []
        seg_masks_reshaped = []
        imgs_used = []

        for img_name in img_names:

            seg_mask = segmenter.segment_sample(img_name)

            seg_area = seg_mask.sum() / seg_mask.size
            if not (min_seg_area <= seg_area <= max_seg_area):
                continue

            seg_masks.append(seg_mask)

            seg_masks_reshaped.append(torch.tensor(resize(seg_mask.astype(float), self.target_shape)))

            imgs_used.append(img_name)

        return seg_masks, torch.stack(seg_masks_reshaped), imgs_used

    def _get_reshaped_activations(self, img_names: Iterable[str]):

        activations = dict()

        for img_name in img_names:

            img_pil = self.img_loader.load_pil_img(self.mscoco_imgs_path, img_name)

            acts_dict, _ = self.activations_extractor.get_bchw_acts_preds_dict(img_pil)

            for l in self.activations_extractor.propagator.layers:
                if l not in activations:
                    activations[l] = []

                acts_temp = acts_dict[l]
                activations[l].append(acts_temp)

        activations = {l: torch.vstack(a) for l, a in activations.items()}

        activations = {l: F.interpolate(a, size=self.target_shape, mode='bilinear') for l, a in activations.items()}

        return activations
