import json
import copy
file = open("train_val_videodatainfo.json",'r')
file_dic = json.load(file)

train_dic = copy.deepcopy(file_dic)
val_dic = copy.deepcopy(file_dic)

ls_txts =  file_dic['sentences']

ls_train_txts =[]
ls_val_txts  =[]

train_file = open('train_videodatainfo.json','w+')
val_file = open('val_videodatainfo.json','w+')

for txt in ls_txts:
    video_id = int(txt['video_id'].replace('video',''))
    new_txt={}
    for key in txt.keys():
        if 'video_id' in key:
            new_txt['clip_name']=txt['video_id']
        else :
            new_txt[key]=txt[key]


    if video_id<6513:
        
        train_file.write(json.dumps(new_txt, ensure_ascii=False)+'\n')
        # ls_train_txts.append(txt)
    elif video_id>=6513 and video_id<=7009:
        val_file.write(json.dumps(new_txt, ensure_ascii=False)+'\n')
