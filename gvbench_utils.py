from hloc import extract_features, match_features, match_dense
from pathlib import Path
from argparse import ArgumentParser

def parser():
      parser = ArgumentParser()
      parser.add_argument('--image_path', type=Path)
      parser.add_argument('--output_path', type=Path)
      parser.add_argument('--features', type=Path)
      parser.add_argument('--pairs', type=Path)
      parser.add_argument('--feature', type=str)
      parser.add_argument('--matcher', type=str)
      
      parser.add_argument('--matching', action='store_true')
      parser.add_argument('--extraction', action='store_true')
      parser.add_argument('--matching_loftr', action='store_true')
      
      return parser.parse_args()


if __name__ == "__main__":
      args = parser()
      
      if args.extraction:
            ### Extract features of each sequences
            if args.feature is not None:
                  features = [args.feature]
            else:
                  features = ['superpoint_max', 'sift', 'disk']
            for feature in features:
                  conf = extract_features.confs[feature]
                  extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path, f'{feature}.h5'))
      
      if args.matching:
            ### Match feature of sequences
            if args.feature is not None:
                  features = [args.feature]
                  if args.matcher is not None:
                        matchers = [args.matcher]
                  
                  for feature in features:
                        if feature == 'superpoint_max':
                              for matcher in matchers:
                                    match_features.main(match_features.confs[matcher], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_{matcher}.h5'))
                        if feature == 'disk':
                              for matcher in matchers:
                                    match_features.main(match_features.confs[matcher], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_{matcher}.h5'))
                        if feature == 'sift':
                              match_features.main(match_features.confs['NN-ratio'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_NN.h5'))
                  
            else:
                  features = ['superpoint_max', 'sift', 'disk']
            
                  for feature in features:
                        if feature == 'superpoint_max':
                              match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_superglue.h5'))
                              match_features.main(match_features.confs['NN-superpoint'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_NN.h5'))
                        if feature == 'disk':
                              match_features.main(match_features.confs['disk+lightglue'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_lightglue.h5'))
                              match_features.main(match_features.confs['NN-ratio'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_NN.h5'))
                        if feature == 'sift':
                              match_features.main(match_features.confs['NN-ratio'], pairs = Path(args.pairs), features= Path(args.features, f'{feature}.h5'), matches= Path(args.output_path, f'{Path(args.pairs).stem}_{feature}_NN.h5'))
      
      if args.matching_loftr:
            ### Match LoFTR of sequences
            match_dense.main(match_dense.confs['loftr'], pairs = Path(args.pairs), image_dir= Path(args.image_path), matches= Path(args.output_path, f'{Path(args.pairs).stem}_loftr.h5'), features= Path(args.features, f'{Path(args.pairs).stem}_loftr_kpts.h5'))
      
      # conf = extract_features.confs[args.feature]
      # extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path))
      
      ### Match feature of sequences
      # match_features.main(match_features.confs['superglue'], pairs = Path(args.pairs), features= Path(args.features), features_ref= Path(args.features_ref), matches= Path(args.output_path))
      
      ### loftr matches
      # conf = match_dense.confs['loftr']
      # match_dense.main(conf, Path(args.pairs), image_dir= Path(args.image_path), export_dir=Path(args.output_path))