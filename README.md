# Local Concept Embeddings for Analysis of Concept Distributions in Vision DNN Feature Spaces

> Mikriukov, Georgii, Gesina Schwalbe, and Korinna Bade. "Local Concept Embeddings for Analysis of Concept Distributions in Vision DNN Feature Spaces." International Journal of Computer Vision (2025).

```
@article{mikriukov2025loce,
  author    = {Georgii Mikriukov and Gesina Schwalbe and Korinna Bade},
  title     = {Local Concept Embeddings for Analysis of Concept Distributions in Vision DNN Feature Spaces},
  journal   = {International Journal of Computer Vision},
  year      = {2025},
  month     = {May},
  doi       = {10.1007/s11263-025-02446-y},
  url       = {https://doi.org/10.1007/s11263-025-02446-y},
  issn      = {1573-1405}
}
```


# LoCE: Local Concept Embeddings and Their Distributions

**Local Concept Embeddings (LoCE)** are a method for analyzing how computer vision DNNs represent object concepts within their latent feature spaces, particularly in complex, real-world scenes. Unlike global concept embedding methods that assign a single vector per category across a dataset, which averages over all samples and erases important context-specific details, **LoCEs generate a distinct embedding for each sample–concept pair**. This enables **fine-grained, context-sensitive analysis** of how models encode objects **relative to both background and surrounding categories**.

**LoCE is designed for use in environments with many interacting objects**, such as autonomous driving, where recognition accuracy is highly dependent on visual context. It supports analysis under conditions of occlusion, interaction, and scene ambiguity.

Each LoCE is computed by optimizing a **compact vector** (shape `C×1×1`) that reconstructs the binary segmentation mask of a target category from the model’s internal activations (shape `C×H×W`). The method uses only activations and external segmentation masks and does **not require any changes to the model**.

### Method Properties

* **Compact**: Each LoCE is a low-dimensional representation (`C×1×1`), efficient for storage, comparison, and retrieval.
* **Context-aware**: LoCEs capture each concept as it appears with its real background and other objects in the scene, not in isolation.
* **Model-agnostic**: Applicable to any pretrained vision model (CNNs, ViTs, etc.) without architectural modifications.
* **Task-agnostic**: Works with models trained for classification, detection, segmentation, or self-supervised tasks.
* **Post-hoc**: No retraining or reconfiguration needed; operates directly on frozen models.
* **Designed for complex scenes**: Tailored for real-world applications with dense object layouts and safety-critical contexts.

### Applications

* **Concept Separability and Purity**: Assess how distinctly the model encodes different object categories.
* **Category Confusion Detection**: Identify overlaps between similar categories (e.g., "bicycle" vs "motorcycle").
* **Sub-concept Discovery**: Uncover unlabeled subtypes or variations within a category (e.g., "flying plane" vs "landed plane").
* **Outlier Detection**: Detect atypical or rare samples that deviate from a category's typical representation.
* **Content-Based Information Retrieval**: Perform efficient, context-aware search using LoCE similarity.
* **Model Comparison**: Evaluate and contrast internal representations across models, layers, or training regimes.

For further details, see the [paper](https://link.springer.com/article/10.1007/s11263-025-02446-y) or [preprint](https://arxiv.org/abs/2311.14435).


### Optimization and Generalization

![3_gcpv_optimization_and_gmm_imgs_compressed.png](./images/3_gcpv_optimization_and_gmm_imgs_compressed.png)

(*left*) **LoCE** optimization for an image-concept pair: **LoCE** *v* represents the optimal convolutional filter weights that *project* Sample's *x* Activations *fₗ(x)* from layer *L* into the Concept Projection Mask *P(v; x)*, aiming to reconstruct the target Concept Segmentation *C* with minimal loss *L(P(v; x), C)*.

(*right*) Distribution of 2D UMAP-reduced **LoCEs** demonstrating the confusion of **car**, **bus**, and **truck** concepts in `DETR.model.encoder.layers.1`. Gaussian Multinomial Mixture (GMM) is fitted to **LoCEs** to highlight the structure. Additionally, some samples from GMM components 2 and 5 are demonstrated.


### Clustering and Distribution Analysis

![umap_detr_model.encoder.layers.1_all_capy_euclidean_one_plot.png](./images/umap_detr_model.encoder.layers.1_all_capy_euclidean_one_plot.png)

Generalization of tested concept **LoCEs** of MS COCO and Capybara Dataset in `DETR.model.encoder.layers.1`:

- 2D UMAP-reduced **LoCEs** of every tested category (*top-left*)
- GMMs fitted for **LoCEs** with regard to their labels (*top-middle*)
- GMMs fitted for all **LoCEs** regardless of their labels (*top-right*)
- **LoCEs** dendrogram with identified clusters (*bottom*)

### Model comparison: Concept Separation

![category_separation.png](./images/category_separation.png)

Pairwise Concept Separation (one-vs-one) in different layers of different models estimated with **LoCEs**.

### Model comparison: Concept-vs-Context Retrieval

![mAP_aggregated_all.png](./images/mAP_aggregated_all.png)

Concept-vs-context, i.e., concept-vs-background, information retrieval with **LoCEs** in complex scenes of MS COCO. mAP@k performance averaged for all tested concepts across models and increasing layer depth.


# Demo 


Download repo:

```bash
git clone https://github.com/continental/local-concept-embeddings
```

Repo Structure:

```
├── data                  <- Dataset downloaders / processed data / data cache.
├── demo                  <- Demonstration files.
├── experiment_outputs    <- (will be created by ./demo/*.ipynb).
├── src                   <- Source files of method.
│   ├── data_structures   <- Data structures: Data loaders, data processors etc.
│   ├── hooks             <- Extraction of activations and gradients.
│   ├── loce              <- Method scripts.
│   └── xai_utils         <- Various utils.
├── Dockerfile            <- Container build instructions.
├── LICENSE               <- License for this project
├── NOTICE                <- Third-party attributions or legal notices
├── README.md             <- This file.
└── requirements.txt      <- Python (exact) dependencies.
```

## Download data

### MS COCO 2017 

Download annotations + validation subset (240 MB + 780 MB), run:

```bash
./data/download_ms_coco_2017val_dataset.sh
```

Data and annotations folder: `./data/mscoco2017val/`

### (Optionally) PASCAL VOC 2012 

Download full dataset (1.9 GB), run:

```bash
./data/download_pascal_voc2012.sh
```

Convert VOC annotations to COCO-style JSON annotations: 

```bash
python ./data/voc2coco.py
```

Data and annotations folder: `./data/voc2012/VOCdevkit/VOC2012/`

> Before running optimization, modify variables with correct paths in [demo/1_optimize.ipynb](demo/1_optimize.ipynb):

```python
imgs_path = "./data/voc2012/VOCdevkit/VOC2012/JPEGImages/"
annots = "./data/voc2012/VOCdevkit/VOC2012/voc2012val_annotations.json"
processed_annots = "./data/voc2012/VOCdevkit/VOC2012/processed/"
```

## Environment setup

### Option 1: Docker + Jupyter

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

2. Build [Dockerfile](Dockerfile) and run:

```
docker build -t loce-gpu .
docker run -d --gpus all -p 8899:8899 -v $PWD:/app --name loce loce-gpu
```

3. Access Jupyter (no password) at: http://localhost:8899 (or your server IP:8899)


### Option 2: Virtual Environment + Jupyter

1. Create & activate venv (optionally), we used Python 3.9.17

```bash
python -m venv test_venv
source ./test_venv/bin/activate
```

2. Install requirements

```bash
pip install -r requirements.txt
```


## Try Jupyter Notebooks

Run [Jupyter Notebooks](demo/):

1. Optimize LoCEs: `./demo/1_optimize.ipynb`
2. Experiments on LoCE distributions (Purity, Separation,  Overlap, and Confusion): `./demo/2_distibution_purity_separation_overlap_confusion.ipynb`
3. Concept-based Retrieval and Outlier Retrieval: `./demo/3_retrieval_and_outliers.ipynb`
2. Sub-concepts inspection: `./demo/4_subconcepts.ipynb`


# Documentation

For further help, see the API-documentation or contact the maintainers.



# License

Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG). All rights reserved.
