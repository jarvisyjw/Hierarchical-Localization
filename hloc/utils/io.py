from typing import Tuple
from pathlib import Path
import numpy as np
import cv2
import h5py
import pickle
import pycolmap

from .parsers import names_to_pair, names_to_pair_old
from .. import logger


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path: Path):
    names = []
    with h5py.File(str(path), 'r', libver='latest') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
        # print('uncertaintylistlens', len(uncertainty))
    if return_uncertainty:
        return p, uncertainty
    return p

def get_descriptors(path: Path, name: str):
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['descriptors']
        p = dset.__array__()
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')


def get_matches(path: Path, name0: str, name1: str, out=None) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    if out is not None:
        matches_score = np.column_stack((matches, scores))
        with open(out, 'w') as f:
            f.write('\n'.join(' '.join(map(str, match)) for match in matches_score))
        f.close()
    return matches, scores

def get_matches_from_path(path: Path, out=None):
    names = list_h5_names(path)
    images1 = []
    images2 = []
    matches = []
    scores = []
    for name in names:
        image1, image2 = name.split('/')
        image1 = image1.replace('-', '/')
        image2 = image2.replace('-', '/')
        images1.append(image1)
        images2.append(image2)
        match, score = get_matches(path, image1, image2, out)
        matches.append(match)
        scores.append(score)
    return images1, images2, matches, scores

def load_colmap_image_poses(Reconstruction: pycolmap.Reconstruction,
                           out = None,
                           ext=".txt"):
    # logger.info(f'Extract poses from ...')
    pose = {}
    # if images is None:
    for  _id , image in Reconstruction.images.items():
        # logger.info(f'None image specified, extract all {len(images)} images.')  
    # for image in images:
        pose[image.name] = (image.qvec, image.tvec)
    if out is not None:
        output = out / f'colmap_image_poses{ext}'
        with open(output, 'w') as f:
            for name, p in pose.items():
                f.write(f'{name} {" ".join(map(str, p[0]))} {" ".join(map(str, p[1]))}\n')
    return pose
