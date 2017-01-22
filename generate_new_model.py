# -*- coding:UTF-8 -*-

import sys,os

caffe_root = '/home/weileilei/caffe/caffe-master/'   

sys.path.insert(0, caffe_root + 'python')

import caffe

os.chdir(caffe_root)

def fillweights(net):
    net.layers
    fillednet = net
    return fillednet

def main():
    net_file_path=caffe_root + 'models/bvlc_alexnet/prune_train_val.prototxt'  
    
    init_model_path=caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    
    net = caffe.Net(net_file_path,init_model_path,caffe.TRAIN)
    
    new_model_path = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet_new.caffemodel'
    
    net.save(new_model_path)

if __name__ == '__main__':
    main()
