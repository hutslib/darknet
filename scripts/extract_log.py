# -*- coding: utf-8 -*-
# 提取训练log，去除不可解析的log后，生成格式化的log，供可视化工具绘制图片
def extract_log(log_file,new_log_file,key_word):
    f = open(log_file)
    train_log = open(new_log_file, 'w')
    for line in f:
        #去除多gpu的同步log
        if 'Syncing' in line:
            continue
        #去除除零错误的log
        if 'nan' in line:
            continue
        if key_word in line:
            train_log.write(line)

    f.close()
    train_log.close()

extract_log('/home/hts/darknet/person_train_log.txt','/home/hts/darknet/person_train_log_loss.txt','images')
extract_log('/home/hts/darknet/person_train_log.txt','/home/hts/darknet/person_train_log_iou.txt','IOU')
