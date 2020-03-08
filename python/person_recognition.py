#!user/bin/python
# _*_ coding: utf-8 _*_

# -------------------------------------------
# # @description: person recognition using yolov2
# # @author: hts
# # @data: 2020-03-07
# # @version: 1.0
# # @github: hutslib
# -------------------------------------------

from darknet_api import yolo_detect

print "suc"
my_yolo = yolo_detect('/home/hts/Pictures/darknet_image_dataset/pic1.jpg')
result = my_yolo.detect_pic()
print result
