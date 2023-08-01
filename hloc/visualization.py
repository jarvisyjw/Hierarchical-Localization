from matplotlib import cm
import random
import numpy as np
import pickle
import pycolmap
from pathlib import Path
import argparse

from .utils.io import get_descriptors, get_keypoints, get_matches, get_matches_from_path
from . import logger

from .utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text, save_plot)
from .utils.io import read_image, list_h5_names
from .utils.viz_3d import plot_camera, init_figure, plot_points, plot_cameras, plot_camera_colmap

def visualize_sfm_2d(reconstruction, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75, out=None, save=False):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)
    
    name2id = {image.name: i for i, image in reconstruction.images.items()}

    if not selected:
        # selected = [image.name for i, image in reconstruction.images.items()]
        image_ids = reconstruction.reg_image_ids()
        selected = random.Random(seed).sample(
                image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[name2id[i]]
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            '''
            bgr
            red for visible
            blue for invisible
            '''
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([reconstruction.points3D[p.point3D_id].track.length()
                           if p.has_point3D() else 1 for p in image.points2D])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array([image.transform_to_image(
                reconstruction.points3D[j].xyz)[-1] for j in p3ids])
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')

        if save:
            logger.info(f'Save image at {str(out)}')
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                save_plot(out)
            else:
                logger.info(f'Image already exists at {str(out)}')



def visualize_loc(results, image_dir, reconstruction=None, db_image_dir=None,
                  selected=[], n=1, seed=0, prefix=None, **kwargs):
    assert image_dir.exists()

    with open(str(results)+'_logs.pkl', 'rb') as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs['loc'].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, min(n, len(queries)))

    if reconstruction is not None:
        if not isinstance(reconstruction, pycolmap.Reconstruction):
            reconstruction = pycolmap.Reconstruction(reconstruction)

    for qname in selected:
        loc = logs['loc'][qname]
        visualize_loc_from_log(
            image_dir, qname, loc, reconstruction, db_image_dir, **kwargs)


def visualize_loc_from_log(image_dir, query_name, loc, reconstruction=None,
                           db_image_dir=None, top_k_db=2, dpi=75):

    q_image = read_image(image_dir / query_name)
    if loc.get('covisibility_clustering', False):
        # select the first, largest cluster if the localization failed
        loc = loc['log_clusters'][loc['best_cluster'] or 0]

    inliers = np.array(loc['PnP_ret']['inliers'])
    mkp_q = loc['keypoints_query']
    n = len(loc['db'])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                         kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc['db'][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert 'keypoints_db' in loc
        assert 'indices_db' in loc
        counts = np.array([
            np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc['db'][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc['db'][db_idx]
            kp_q = mkp_q[loc['indices_db'] == db_idx]
            kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
            inliers_db = inliers[loc['indices_db'] == db_idx]

        db_image = read_image((db_image_dir or image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f'inliers: {sum(inliers_db)}/{len(inliers_db)}'

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)

def visualize_match_cross_view(image0: str, image1: str, 
                              match_path: Path,
                              feature0_path: Path,
                              feature1_path: Path,
                              database_image0: Path,
                              database_image1: Path,
                              out: Path,
                              dpi=75,
                              save=True):
    matches, _ = get_matches(match_path, image0, image1)
    matches0 = matches[:,0]
    matches1 = matches[:,1]

    keypoint0 = get_keypoints(feature0_path, image0)
    keypoint1 = get_keypoints(feature1_path, image1)

    kp_0 = keypoint0[matches0]
    kp_1 = keypoint1[matches1]

    image_0 = read_image(database_image0 / image0)
    image_1 = read_image(database_image1 / image1)

    plot_images([image_0, image_1], dpi=dpi)
    plot_matches(kp_0, kp_1, a = 0.1)
    add_text(0, image0)
    add_text(1, image1)
    if save:
        logger.info(f'Save image at {str(out)}')
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            save_plot(out)
        else:
            logger.info(f'Image already exists at {str(out)}')


def visualize_match_from_pair(image0: str, image1: str, 
                              match_path: Path,
                              feature_path: Path,
                              database_image: Path,
                              out= None,
                              dpi=75,
                              save=True,
                              kpts=False):
    matches, _ = get_matches(match_path, image0, image1)
    matches0 = matches[:,0]
    matches1 = matches[:,1]

    keypoint0 = get_keypoints(feature_path, image0)
    keypoint1 = get_keypoints(feature_path, image1)

    kp_0 = keypoint0[matches0]
    kp_1 = keypoint1[matches1]

    image_0 = read_image(database_image / image0)
    image_1 = read_image(database_image / image1)

    plot_images([image_0, image_1], dpi=dpi)
    plot_matches(kp_0, kp_1, a = 0.1)
    add_text(0, image0)
    add_text(1, image1)
    add_text(0, f'num_matches: {len(kp_0)}', pos=(0.01, 0.01))
    name0 = image0.replace("/", "-").strip(".jpg")
    name1 = image1.replace("/", "-").strip(".jpg")
    name = f'{name0}_{name1}'
    if save:
        out = out / f'{name}.png'
        logger.info(f'Save image at {str(out)}')
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            save_plot(out)
        else:
            logger.info(f'Image already exists at {str(out)}')

def visualize_matches_from_path_cross_view(match_path: Path,
                                feature0_path: Path,
                                feature1_path: Path,
                                database_image0: Path,
                                database_image1: Path,
                                out: Path,
                                names1=None,
                                names2=None,
                                pairs = None,
                                dpi=75,
                                save=True):
    if pairs is None:
        logger.info('No pairs specified, visualize all pairs')
        names1, names2, pairs, _  = get_matches_from_path(match_path)
    
    assert len(names1) == len(names2) == len(pairs)
    for image0, image1 , pair in zip(names1, names2, pairs):
        logger.info(f'Visualizing matches for pair {image0} and {image1}')
        '''
        /workspace/outputs/NUS/BD/t0/debug/Left/006689.png_Left/005152.png.png
        /workspace/outputs/NUS/BD/t0/debug/Left/ 006689 / Left_005152.png
        '''
        match_pair = image0.strip('.png') + '/' + image1.replace('/','-')
        match_pair = match_pair.strip('.png')
        visualize_match_cross_view(image0, image1, match_path, feature0_path , feature1_path, database_image0, database_image1, out / f'{match_pair}.png', dpi, save)

def visualize_matches_from_path(match_path: Path,
                                feature_path: Path,
                                database_image: Path,
                                out: Path,
                                names1=None,
                                names2=None,
                                pairs = None,
                                dpi=75,
                                save=True):
    if pairs is None:
        logger.info('No pairs specified, visualize all pairs')
        names1, names2, pairs, _  = get_matches_from_path(match_path)
    
    assert len(names1) == len(names2) == len(pairs)
    for image0, image1 , pair in zip(names1, names2, pairs):
        logger.info(f'Visualizing matches for pair {image0} and {image1}')
        '''
        /workspace/outputs/NUS/BD/t0/debug/Left/006689.png_Left/005152.png.png
        /workspace/outputs/NUS/BD/t0/debug/Left/ 006689 / Left_005152.png
        '''
        match_pair = image0.strip('.png') + '/' + image1.replace('/','-')
        match_pair = match_pair.strip('.png')
        visualize_match_from_pair(image0, image1, match_path, feature_path , database_image, out / f'{match_pair}.png', dpi, save)
 
def visualize_keypoints_from_file(image: str, 
                                  feature_path: Path,
                                  database_image: Path, 
                                  out: Path,
                                  dpi=75,
                                  save=True):
    keypoint = get_keypoints(feature_path, image)
    image_0 = read_image(database_image / image)
    plot_images([image_0], dpi=dpi)
    plot_keypoints([keypoint])
    add_text(0, image)
    if save:
        logger.info(f'Save image at {str(out)}')
        if not out.exists():
            out.parent.parent.mkdir(parents=True, exist_ok=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            save_plot(out)
        else:
            logger.info(f'Image already exists at {str(out)}')

def visualize_keypoints_from_path(feature_path: Path,
                                  database_image: Path, 
                                  out: Path,
                                  images=None,  
                                  dpi=75,
                                  save=True):
    if images is None:
        images = list_h5_names(feature_path)
    
    for image in images:
        if not out.exists():
            out.mkdir(parents=True, exist_ok=True)
        visualize_keypoints_from_file(image, feature_path, database_image, out / image, dpi, save)

def parser():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument("--mode", type=str, default="match", help="mode: match, keypoint, visible")
    parser.add_argument("--match_path", type=str, default="/workspace/outputs/Nerf_synthetic/train/matches.h5", help="path to match file")
    parser.add_argument("--feature_path", type=str, default="/workspace/outputs/Nerf_synthetic/train/features.h5", help="path to feature file")
    parser.add_argument("--database_image", type=str, default="/workspace/dataset/Nerf/nerf_synthetic/lego/train", help="path to database image")
    parser.add_argument("--out", type=str, default="/workspace/outputs/Nerf_synthetic/train/matches", help="path to output image")
    parser.add_argument("--colmap_model", type=str, default="/workspace/outputs/Nerf_synthetic/train/sfm", help="path to colmap model")
    return parser.parse_args()

def main():
    args = parser()
    if args.mode == "match":
        visualize_matches_from_path(Path(args.match_path), Path(args.feature_path), Path(args.database_image), Path(args.out))
    if args.mode == "keypoint":
        visualize_keypoints_from_path(Path(args.feature_path), Path(args.database_image), Path(args.out))
    if args.mode == "visible":
        model = pycolmap.Reconstruction(Path(args.colmap_model))
        output = Path(args.out)
        for i, image in model.images.items():
            visualize_sfm_2d(model, Path(args.database_image), color_by='visibility', selected=[image.name],
                            out=Path( output / f'{image.name}'), save=True)

if __name__ == '__main__':
    main()