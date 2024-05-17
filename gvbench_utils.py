from hloc import extract_features
from pathlib import Path
from argparse import ArgumentParser

def parser():
      parser = ArgumentParser()
      parser.add_argument('--image_path', type=str, required=True)
      parser.add_argument('--output_path', type=str, required=True)
      parser.add_argument('--feature', type=str, required=True)
      return parser.parse_args()




if __name__ == "__main__":
      args = parser()
      ### Extract features of each sequences
      conf = extract_features.confs[args.feature]
      extract_features.main(conf, Path(args.image_path), feature_path = Path(args.output_path))