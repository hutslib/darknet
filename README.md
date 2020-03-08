<!-- TOC -->

- [【1】安装opencv](#1安装opencv)
- [【2】安装 cuda](#2安装-cuda)
- [【3】安装cudnn](#3安装cudnn)
- [【4】darknet](#4darknet)
- [【5】YOLO训练自己的数据集](#5yolo训练自己的数据集)
    - [· 修改darknet文件](#·-修改darknet文件)
    - [· labelImg制作自己的数据集](#·-labelimg制作自己的数据集)
    - [· 划分训练集与测试集](#·-划分训练集与测试集)
    - [· 训练](#·-训练)
    - [· 测试训练模型](#·-测试训练模型)
    - [· 训练过程中的输出参数含义](#·-训练过程中的输出参数含义)
- [Darknet](#darknet)

<!-- /TOC -->
# 【1】安装opencv
官方教程网址 https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html
	##python2（我只尝试了python2)
	sudo apt-get install python-opencv
	sudo apt-get install libopencv-dev
	## python3
	pip3 install python-opencv

# 【2】安装 cuda 
1. 下载 
	cuda下载网址https://developer.nvidia.com/cuda-toolkit-archive
	下载cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
2. 安装
	```
	sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
	sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
	sudo apt-get update
	sudo apt-get install cuda-9-0 
	sudo reboot
	```
3. 添加环境变量
	```
	sudo gedit ~/.bashrc
	##添加如下内容
	export CUDA_HOME=/usr/local/cuda-9.0
	export PATH=/usr/local/cuda-9.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
	```
4. 检查是否安装成功 
==如果遇到提示不要执行 这一句 sudo apt install nvidia-cuda-toolkit==
	```
	nvcc -V
	```
# 【3】安装cudnn
1. 下载cudnn-9.0-linux-x64-v7.tgz
	下载网址：https://developer.nvidia.com/rdp/cudnn-archive
2. 安装
	```
	sudo tar -xzvf cudnn-9.0-linux-x64-v7.tgz
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda./darknet/lib64/libcudnn*
	```
3. 添加环境变量
	```
	sudo gedit ~/.bashrc
	#添加以下内容
	export 	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
	export PATH=$PATH:/usr/local/cuda/bin
	```
# 【4】darknet
官网教程 https://pjreddie.com/darknet/install/
1. 下载darknet
	```
	git clone https://github.com/pjreddie/darknet.git
	```
2. 修改makefile
修改的内容有GPU、CUDNN、 OPENCV、  ARCH、 NVCC、  COMMON+= -DGPU 、 LDFLAGS+= 这几行根据自己的电脑配置做适当的修改。
```
GPU=1
CUDNN=1
OPENCV=1
OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_61,code=[sm_61,compute_61]

#ARCH= -gencode arch=compute_30,code=sm_30 \
#      -gencode arch=compute_35,code=sm_35 \
#      -gencode arch=compute_50,code=[sm_50,compute_50] \
#     -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52
中间无修改省略
#NVCC=nvcc 
NVCC=/usr/local/cuda-9.0/bin/nvcc
中间无修改省略
ifeq ($(GPU), 1) 
#COMMON+= -DGPU -I/usr/local/cuda/include/
COMMON+= -DGPU -I/usr/local/cuda-9.0/include/
CFLAGS+= -DGPU
#LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LDFLAGS+= -L/usr/local/cuda-9.0/lib64 -lcuda -lcudart -lcublas -lcurand
endif
以下无修改省略
```
3. 编译
	```
	cd darknet
	make
	```
4.  测试
	```
	./darknet
	#输出如下编译成功
	usage: ./darknet <function>
	```
5. 测试opencv
 	```
	 ./darknet imtest data/eagle.jpg
	 ```
# 【5】YOLO训练自己的数据集
## · 修改darknet文件

1. 在darknet 文件目录下面找到或建立 /darknet/training/image文件夹存放待标注图片
2. 在darknet/data文件夹下修改创建voc.names文件.在该文件中写入所有类别的名称，每一类占一行。
3. 在darknet/cfg文件夹下，创建并修改voc.data打开，修改其中的内容
	```
	classes= 1          #训练数据的类别数目，我的为1
	train  = <path to darknet>/training/train_set.txt  #划分训练集和测试时存储路径的txt文件
	valid  = <path to darknet>/training/test_set.txt        #划分训练集和测试时存储路径的txt文件
	names = <path to darknet>/data/voc.names                    #创建的names文件路径
	backup = <path to darknet>/backup                      #这是训练得到的model的存放目录，建议自己修改。
	```
4. 应用yolov2_voc.cfg网络来训练你的数据，需要修改这个文件中的一些内容。
	```
	[region]层中classes改成你的类别数，我这里只检测挥手，所以我改成了classes=1.
	[region]层上方的[convolution]层中，filters的数量要根据[region]写的参数更改成（classes+coords+1）*num。我这里改成了(1+4+1)*5=30.
	
	##参数介绍
	classes：类别数量 
	coords：BoundingBox的tx,ty,tw,th，tx与ty是相对于左上角的gird，同时是当前grid的比例，tw与th是宽度与高度取对数 
	num：每个grid预测的BoundingBox个数 
	```
## · labelImg制作自己的数据集
0. 从视频获取图片，我上传了代码在我的github上面
	```
	python getframe.py --video_path ~/darknet/video/IMG_1547.avi --save_path ~/darknet/training/image
	```
2. 安装labelImg，安装完labelImg后，在labelImg/data/predefined_classes.txt中写入label的类别

	因为电脑上有qt5所以只能用python3 + qt5 版本否则会报错，RuntimeError: the PyQt4.QtCore and PyQt5.QtCore modules both wrap the QObject class，这个错误的原因是qt4 也有qt5， 解决办法是使用python3 + qt5， 或者卸载qt5， 但我卸载qt5 的时候ros相关的包也会卸载掉，所以选择使用python3+qt5)
	```
	# python3+qt5 
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    cd labelImg
    make qt5py3
    python3 labelImg.py # 打开labelImg
	```
3. labelImg 的使用（具体可参考labelImg的README，我觉得README里面写的很好了）
	更改 Open Dir 为 /darknet/training/image
	更改 Change Save Dir 为 /darknet/training/label
	将软件默认的Pascal/VOC更改为YOLO
	点击Create\nRectBox 圈出待标注物体, 输入类别标签后保存
	通过Next Image 和 Prev Image 来更换图片
    快捷键参考README
## · 划分训练集与测试集
1. 写了一个代码在我的github上面，需要修改路径
	```
	python pick_train&test_jpg.py
	```
## · 训练
0. 在训练前要把/darknet/training/label/文件夹下面的所有文件拷贝到/darknet/training/image/文件夹下面
1.  官网参考教程: https://pjreddie.com/darknet/yolov2/
	```
	cd darknet
	wget https://pjreddie.com/media/files/darknet19_448.conv.23
	./darknet detector train cfg/voc.data cfg/yolov2-voc.cfg darknet19_448.conv.23
	# 暂停后再继续
	./darknet detector train cfg/voc.data cfg/yolov2-voc.cfg backup/yolov2-voc.backup
	```
## · 测试训练模型
1.  在我的github上面增添了检测的一些api可以根据需要调用，文件名为darknet_api.py，
	```
	#注意修改lib = CDLL("~/darknet/libdarknet.so", RTLD_GLOBAL)
	#    def __init__(self, img_path, cfg_path = '~/darknet/cfg/yolov2-voc.cfg', weight_path = '~/Documents/darknet_backup/yolov2-voc_900.weights', meta_path = '~/darknet/cfg/voc.data'):
	#路径
	#以及    my_detector = yolo_detect('/home/hts/Pictures/my_pic_darknet/37.jpg')
	```
	调用api我写了一个例子叫person_recognition.py
## · 训练过程中的输出参数含义
```
#训练过程中终端输出
Loaded: 0.000034 seconds
Region Avg IOU: 0.807066, Class: 1.000000, Obj: 0.950778, No Obj: 0.003734, Avg Recall: 1.000000,  count: 1
6231: 0.907933, 0.625444 avg, 0.001000 rate, 0.033331 seconds, 6231 images
```
1. 每一次输出的结果表明一个batch训练结束，batch的大小在yolov2-voc.cfg中设定，例如每轮迭代会从所有训练集里随机抽取 batch = 64 个样本参与训练，所有这些 batch 个样本又被均分为 subdivision = 8 次送入网络参与训练，以减轻内存占用的压力。我这里batch 和subdivision都是1
2. 最后一行
	```
	6231: 0.907933, 0.625444 avg, 0.001000 rate, 0.033331 seconds, 6231 images
	```
数字     | 含义
-------- | -----
6231 | 当前迭代次数
0.907933  | 总体的损失
0.625444 | 平均损失（越低越好0.01以下再考虑停止）
0.001000rate|当前的学习率，是在yolov2-voc.cfg文件中定义的
0.033331 seconds| 当前batch训练完花费的总时间
6231 images| 到目前为止，参与训练的图片的总量
3.  IOU含义 
	IOU代表预测的矩形框和真实目标的交集与并集之比
	$$
	IOU = \frac{AreaOfOverlap}{AreaOfUnion}
	$$ 
	IOU = 100% 时表示预测与实际完全重合，我们需要优化IOU。
5. IOU 行输出含义	
	```
	Region Avg IOU: 0.807066, Class: 1.000000, Obj: 0.950778, No Obj: 0.003734, Avg Recall: 1.000000,  count: 1
	```
数字     | 含义
-------- | -----
Region Avg IOU | 在当前subdivision内的图片的平均IOU
Class| 标注物体分类的正确率越接近1越好
Obj|越接近1越好
No Obj|越小越好，但不为0
Avg Recall|在所有subdivision图片中检测出的正样本与实际的正样本的比值

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


