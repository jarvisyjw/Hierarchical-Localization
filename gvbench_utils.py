from hloc import extract_features, match_features, match_dense
from pathlib import Path

import yaml
import argparse

def parser():
      parser = argparse.ArgumentParser()
      parser.add_argument('config', type=str, help='Path to the config file')
      # parser.add_argument('--image_path', type=Path)
      # parser.add_argument('--output_path', type=Path)
      # parser.add_argument('--features', type=Path)
      # parser.add_argument('--pairs', type=Path)
      # parser.add_argument('--feature', type=str)
      # parser.add_argument('--matcher', type=str)
      
      parser.add_argument('--matching', action='store_true')
      parser.add_argument('--extraction', action='store_true')
      parser.add_argument('--matching_loftr', action='store_true')
      
      args = parser.parse_args()
      
      def dict2namespace(config):
            namespace = argparse.Namespace()
            for key, value in config.items():
                  if isinstance(value, dict):
                        new_value = dict2namespace(value)
                  else:
                        new_value = value
                  setattr(namespace, key, new_value)
            return namespace
    
      # parse config file
      with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
      config = dict2namespace(config)
      
      return args, config


if __name__ == "__main__":
      args, cfg = parser()
      
      if args.extraction:
            ext = cfg.extraction
            ### Extract features of each sequences
            if ext.features is not None:
                  features = [ext.features]
            else:
                  features = ['superpoint_max', 'sift', 'disk']
            for feature in features:
                  conf = extract_features.confs[feature]
                  image_path = Path(ext.image_path)
                  output_path = Path(ext.output_path, f'{feature}.h5')
                  extract_features.main(conf, image_path, feature_path = output_path)
      
      if args.matching:
            ### Match feature of sequences
            mat = cfg.matching
            pairs = Path(mat.pairs)
            if mat.feature is not None:
                  features = [mat.feature]
                  if mat.matcher is not None:
                        matchers = [mat.matcher]
                  
                  for feature in features:
                        feature_path = Path(mat.feature_path, f'{feature}.h5')
                        if feature == 'superpoint_max':
                              for matcher in matchers:
                                    match_features.main(match_features.confs[matcher], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_{matcher}.h5'))
                        if feature == 'disk':
                              for matcher in matchers:
                                    match_features.main(match_features.confs[matcher], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_{matcher}.h5'))
                        if feature == 'sift':
                              match_features.main(match_features.confs['NN-ratio'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_NN.h5'))
                  
            else:
                  features = ['superpoint_max', 'sift', 'disk']
                  for feature in features:
                        feature_path = Path(mat.feature_path, f'{feature}.h5')
                        if feature == 'superpoint_max':
                              match_features.main(match_features.confs['superglue'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_superglue.h5'))
                              match_features.main(match_features.confs['NN-superpoint'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_NN.h5'))
                        if feature == 'disk':
                              match_features.main(match_features.confs['disk+lightglue'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_lightglue.h5'))
                              match_features.main(match_features.confs['NN-ratio'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_NN.h5'))
                        if feature == 'sift':
                              match_features.main(match_features.confs['NN-ratio'], pairs = pairs, features= feature_path, matches= Path(mat.output_path, f'{pairs.stem}_{feature}_NN.h5'))
      
      if args.matching_loftr:
            ### Match LoFTR of sequences
            pairs = Path(cfg.matching_loftr.pairs)
            image_path = Path(cfg.matching_loftr.image_path)
            matches = Path(cfg.matching_loftr.matches, f'{pairs.stem}_loftr.h5')
            features = Path(cfg.matching_loftr.features, f'{pairs.stem}_loftr_kpts.h5') 
            match_dense.main(match_dense.confs['loftr'], pairs = pairs, image_dir= image_path, matches= matches, features= features)
      
      # conf = extract_features.confs[args.feature]
      # extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path))
      
      ### Match feature of sequences
      # match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.features), features_ref= Path(args.features_ref), matches= Path(args.output_path))
      
      ### loftr matches
      # conf = match_dense.confs['loftr']
      # match_dense.main(conf, Path(args.pairs), image_dir= Path(args.image_path), export_dir=Path(args.output_path))