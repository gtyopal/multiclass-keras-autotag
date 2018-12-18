# coding:utf-8
import argparse
import os
from classifier.text_classifier import TextClassifier
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

""" Command or IDE run instruction:
if cmd = False, then run on IDE
if cmd = True, then run on Command
Command run Usage: python predict.py --conf config/spam.json """

cmd = False
conf = "config/en.json"

if cmd:
    parse = argparse.ArgumentParser()
    parse.add_argument("--conf", help="config file", required=True)
    args = parse.parse_args()
    conf = args.conf

# init model
model = TextClassifier(conf_path=conf, ispredict=1)

# predict
text = "pwpn downgrade category i 'm trypng to bupld an pwpn aii where you can downgrade to any pwpn u want but pdk what the category of my aii ps gone be"

y_pred = model.predict(text)

max_prob = max(y_pred[0])
max_index = list(y_pred[0]).index(max_prob)
print(max_index)
print(model.id2category[max_index])
print(max_prob)
