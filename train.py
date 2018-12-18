# coding:utf-8
import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from classifier.text_classifier import TextClassifier

""" Command or IDE run instruction:
if cmd = False, then run on IDE
if cmd = True, then run on Command
Command run Usage: python train.py --conf config/en.json --gpu 0"""

cmd = False
conf = "config/en.json"

if cmd:
    parse = argparse.ArgumentParser()
    parse.add_argument("--conf", help="config file", required=True)
    args = parse.parse_args()
    conf = args.conf


# init model
model = TextClassifier(conf_path=conf, ispredict=0)

# training
model.train()
