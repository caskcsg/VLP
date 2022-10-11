ls_now = []
lines = open('train_list_jsfusion.txt','r').readlines()

import random 
random.shuffle(lines)

val_file = open('mvp_val_list.txt','w+')
train_file = open('mvp_train_list.txt','w+')
for line in lines[:500]:
    val_file.write(line.strip()+'\n')
for line in lines[500:]:
    train_file.write(line.strip()+'\n')

