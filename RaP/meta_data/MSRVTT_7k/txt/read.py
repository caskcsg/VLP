import json
import copy

train_file = open('train_videodatainfo.json','r')
val_file = open('val_videodatainfo.json','r')

with open(filename, "r") as f:
    ls =  [json.loads(l.strip("\n")) for l in f.readlines()]

# val_json = json.load(val_file)
# train_json = json.load(train_file)
import ipdb;ipdb.set_trace()
