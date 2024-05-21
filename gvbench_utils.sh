#!/bin/bash

# First extract features
echo "Extracting features..."
python gvbench_utils.py --extraction --image_path datasets/GV-Bench/release/images --output_path datasets/GV-Bench/release/features

# Then match the features
# day sequence
echo "Matching features for day sequence..."
python gvbench_utils.py --matching --pairs datasets/GV-Bench/release/gt/day.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

# night sequence
echo "Matching features for night sequence..."
python gvbench_utils.py --matching --pairs datasets/GV-Bench/release/gt/night.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

# season sequence
echo "Matching features for season sequence..."
python gvbench_utils.py --matching --pairs datasets/GV-Bench/release/gt/season.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

# weather sequence
echo "Matching features for weather sequence..."
python gvbench_utils.py --matching --pairs datasets/GV-Bench/release/gt/weather.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

# Match dense using LoFTR
echo "Matching featurs by LoFTR..."

echo "Matching features for day sequence.."
python gvbench_utils.py --matching_loftr --pairs datasets/GV-Bench/release/gt/day.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

echo "Matching features for night sequence.."
python gvbench_utils.py --matching_loftr --pairs datasets/GV-Bench/release/gt/night.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

echo "Matching features for season sequence.."
python gvbench_utils.py --matching_loftr --pairs datasets/GV-Bench/release/gt/season.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches

echo "Matching features for weather sequence.."
python gvbench_utils.py --matching_loftr --pairs datasets/GV-Bench/release/gt/weather.txt --features datasets/GV-Bench/release/features --output_path datasets/GV-Bench/release/matches