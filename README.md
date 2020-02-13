# Polisis_Benchmark
Reproducing state-of-the-art results

This repo is our effort to reproduce Polisis results on privacy policy classification based on their paper: https://arxiv.org/abs/1802.02561 

# Setup instructions
1. Setup a virtual environment using any tool (e.g., conda) and activate it: conda -n privacy_policy python=3.6 source activate privacy_policy
2. Install dependecies from the requirement file: pip install -r requirement.txt
3. install NLTK tokenizer: python -m nltk.downloader punkt

# Usage instructions
To run the experiment: python -u cnn_multi_label_classifier.py

Parameters can be found in args.py

Important Note: By default the code will use GloVe embeddings. Due to licesing the in-domain embeddings can be provided only upon request.
