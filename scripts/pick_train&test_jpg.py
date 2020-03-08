#! /usr/bin/env python
# -*- encoding: UTF-8 -*-
import os

def main():

    img_dir = "/home/hts/darknet/training/image"
    train_txt = open('/home/hts/darknet/training/path/train_set.txt', 'w')
    test_txt = open('/home/hts/darknet/training/path/test_set.txt', 'w')
    img_list = os.listdir(img_dir)
    for i in range(0, len(img_list)):
        path = os.path.join(img_dir, img_list[i])
        if i % 5 == 0:
            test_txt.writelines(path + '\n')
        else:
            train_txt.writelines(path + '\n')
    train_txt.close()
    test_txt.close()

if __name__ == '__main__':
    main()
