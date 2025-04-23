# hloc - the hierarchical localization toolbox

This is `hloc`, a modular toolbox for state-of-the-art 6-DoF visual localization. It implements [Hierarchical Localization](https://arxiv.org/abs/1812.03506), leveraging image retrieval and feature matching, and is fast, accurate, and scalable. This codebase combines and makes easily accessible years of research on image matching and Structure-from-Motion. Refer to the original [README](README_old.md) for basic installation. 


## Customized Pipeline for feature extraction and matching
This mainly serves for the easy usage of [GV-Bench](https://github.com/jarvisyjw/GV-Bench). The customized pipeline includes 6 keypoints detection and matching used in GV-Bench:

<p align="center">
   <a href="https://arxiv.org/abs/2407.11736"><img src="./doc/matching-methods.png" width="60%"/></a>
  <br /><em>Matching Methods Evaluated in GV-Bench</em>
</p>

### Usage
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

All dependencies are listed in `requirements.txt`. **Starting with `hloc-v1.3`, installing COLMAP is not required anymore.** This repository includes external local features as git submodules – don't forget to pull submodules with `git submodule update --init --recursive`.

We also provide a Docker image:
```bash
docker build -t hloc:latest .
docker run -it --rm -p 8888:8888 hloc:latest  # for GPU support, add `--runtime=nvidia`
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

## General pipeline

The toolbox is composed of scripts, which roughly perform the following steps:

1. Extract local features, like [SuperPoint](https://arxiv.org/abs/1712.07629) or [DISK](https://arxiv.org/abs/2006.13566), for all database and query images
2. Build a reference 3D SfM model
   1. Find covisible database images, with retrieval or a prior SfM model
   2. Match these database pairs with [SuperGlue](https://psarlin.com/superglue/) or the faster [LightGlue](https://github.com/cvg/LightGlue)
   3. Triangulate a new SfM model with COLMAP
3. Find database images relevant to each query, using retrieval
4. Match the query images
5. Run the localization
6. Visualize and debug

The localization can then be evaluated on [visuallocalization.net](https://www.visuallocalization.net/) for the supported datasets. When 3D Lidar scans are available, such as for the indoor dataset InLoc, step 2. can be skipped.

Strcture of the toolbox:

- `hloc/*.py` : top-level scripts
- `hloc/extractors/` : interfaces for feature extractors
- `hloc/matchers/` : interfaces for feature matchers
- `hloc/pipelines/` : entire pipelines for multiple datasets

`hloc` can be imported as an external package with `import hloc` or called from the command line with:
```bash
python -m hloc.name_of_script --arg1 --arg2
```

## Tasks

We provide step-by-step guides to localize with Aachen, InLoc, and to generate reference poses for your own data using SfM. Just download the datasets and you're reading to go!

### Aachen – outdoor localization

Have a look at [`pipeline_Aachen.ipynb`](https://nbviewer.jupyter.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_Aachen.ipynb) for a step-by-step guide on localizing with Aachen. Play with the visualization, try new local features or matcher, and have fun! Don't like notebooks? You can also run all scripts from the command line.

<p align="center">
  <a href="https://nbviewer.jupyter.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_Aachen.ipynb"><img src="doc/loc_aachen.svg" width="70%"/></a>
</p>

### InLoc – indoor localization

The notebook [`pipeline_InLoc.ipynb`](https://nbviewer.jupyter.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_InLoc.ipynb) shows the steps for localizing with InLoc. It's much simpler since a 3D SfM model is not needed.

<p align="center">
  <a href="https://nbviewer.jupyter.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_InLoc.ipynb"><img src="doc/loc_inloc.svg" width="70%"/></a>
</p>

### SfM reconstruction from scratch

We show in [`pipeline_SfM.ipynb`](https://nbviewer.jupyter.org/github/cvg/Hierarchical-Localization/blob/master/pipeline_SfM.ipynb) how to run 3D reconstruction for an unordered set of images. This generates reference poses, and a nice sparse 3D model suitable for localization with the same pipeline as Aachen.

## Results

- Supported local feature extractors: [SuperPoint](https://arxiv.org/abs/1712.07629), [DISK](https://arxiv.org/abs/2006.13566), [D2-Net](https://arxiv.org/abs/1905.03561), [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), and [R2D2](https://arxiv.org/abs/1906.06195).
- Supported feature matchers: [SuperGlue](https://arxiv.org/abs/1911.11763), its faster follow-up [LightGlue](https://github.com/cvg/LightGlue), and nearest neighbor search with ratio test, distance test, and/or mutual check. hloc also supports dense matching with [LoFTR](https://github.com/zju3dv/LoFTR).
- Supported image retrieval: [NetVLAD](https://arxiv.org/abs/1511.07247), [AP-GeM/DIR](https://github.com/naver/deep-image-retrieval), [OpenIBL](https://github.com/yxgeee/OpenIBL), and [MegaLoc](https://github.com/gmberton/MegaLoc).

Using NetVLAD for retrieval, we obtain the following best results:

| Methods                                                      | Aachen day         | Aachen night       | Retrieval      |
| ------------------------------------------------------------ | ------------------ | ------------------ | -------------- |
| [SuperPoint + SuperGlue](https://www.visuallocalization.net/details/10931/) | 89.6 / 95.4 / 98.8 | 86.7 / 93.9 / 100  | NetVLAD top 50 |
| [SuperPoint + NN](https://www.visuallocalization.net/details/10866/) | 85.4 / 93.3 / 97.2 | 75.5 / 86.7 / 92.9 | NetVLAD top 30 |
| D2Net (SS) + NN                                              | 84.6 / 91.4 / 97.1 | 83.7 / 90.8 / 100  | NetVLAD top 30 |

| Methods                                                      | InLoc DUC1         | InLoc DUC2         | Retrieval      |
| ------------------------------------------------------------ | ------------------ | ------------------ | -------------- |
| [SuperPoint + SuperGlue](https://www.visuallocalization.net/details/10936/) | 46.5 / 65.7 / 78.3 | 52.7 / 72.5 / 79.4 | NetVLAD top 40 |
| [SuperPoint + SuperGlue (temporal)](https://www.visuallocalization.net/details/10937/) | 49.0 / 68.7 / 80.8 | 53.4 / 77.1 / 82.4 | NetVLAD top 40 |
| [SuperPoint + NN](https://www.visuallocalization.net/details/10896/) | 39.9 / 55.6 / 67.2 | 37.4 / 57.3 / 70.2 | NetVLAD top 20 |
| D2Net (SS) + NN                                              | 39.9 / 57.6 / 67.2 | 36.6 / 53.4 / 61.8 | NetVLAD top 20 |

Check out [visuallocalization.net/benchmark](https://www.visuallocalization.net/benchmark) for more details and additional baselines.

## Supported datasets

We provide in [`hloc/pipelines/`](./hloc/pipelines) scripts to run the reconstruction and the localization on the following datasets: Aachen Day-Night (v1.0 and v1.1), InLoc, Extended CMU Seasons, RobotCar Seasons, 4Seasons, Cambridge Landmarks, and 7-Scenes. For example, after downloading the dataset [with the instructions given here](./hloc/pipelines/Aachen#installation), we can run the Aachen Day-Night pipeline with SuperPoint+SuperGlue using the command:
```bash
python -m hloc.pipelines.Aachen.pipeline [--outputs ./outputs/aachen]
```

## BibTex Citation

If use any of the tools provided here, please consider citing both [Hierarchical Localization](https://arxiv.org/abs/1812.03506) and [SuperGlue](https://arxiv.org/abs/1911.11763), and perhaps [GV-Bench](https://arxiv.org/abs/2407.11736).

```
@inproceedings{sarlin2019coarse,
  title     = {From Coarse to Fine: Robust Hierarchical Localization at Large Scale},
  author    = {Paul-Edouard Sarlin and
               Cesar Cadena and
               Roland Siegwart and
               Marcin Dymczyk},
  booktitle = {CVPR},
  year      = {2019}
}

@inproceedings{sarlin2020superglue,
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  booktitle = {CVPR},
  year      = {2020},
}

@article{yu2024gv,
  title={GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection},
  author={Yu, Jingwen and Ye, Hanjing and Jiao, Jianhao and Tan, Ping and Zhang, Hong},
  journal={arXiv preprint arXiv:2407.11736},
  year={2024}
}
```

## Acknowledgement
Thanks for the original [[HLoc]](./README_old.md).
