# Wide-ResNet CIFAR-10
from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def conv_factory(bottom, ks, n_out, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=n_out, pad=pad,
                         bias_filler=dict(type='constant', value=0), weight_filler=dict(type='msr'))
    return conv

def residual_block(bottom, num_filters, stride=1, diffInputOutput=False):
    batch_norm0 = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale0 = L.Scale(batch_norm0, bias_term=True, in_place=True)
    relu0 = L.ReLU(scale0, in_place=True)
    conv1 = conv_factory(relu0, 3, num_filters, stride, 1)
    batch_norm1 = L.BatchNorm(conv1, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale1 = L.Scale(batch_norm1, bias_term=True, in_place=True)
    relu1 = L.ReLU(scale1, in_place=True)
    conv2 = conv_factory(relu1, 3, num_filters, 1, 1)
    if diffInputOutput == False:
        add = L.Eltwise(conv2, bottom, operation=P.Eltwise.SUM)
        return add
    else:
        conv_shortcut = conv_factory(bottom, 1, num_filters, stride, 0)
        add = L.Eltwise(conv2, conv_shortcut, operation=P.Eltwise.SUM)
        return add

def resnet_cifar(depth=16, widen_factor=1, classes=10):
    assert (depth-4) % 6 == 0
    n = (depth - 4) / 6

    data, label = L.Data(ntop=2, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    residual = conv_factory(None, 3, 16, 1, 1)

    residual = residual_block(residual, 16*widen_factor, 1, True)
    for i in xrange(n - 1):
        residual = residual_block(residual, 16*widen_factor)

    residual = residual_block(residual, 32*widen_factor, 2, True)
    for i in xrange(n - 1):
        residual = residual_block(residual, 32*widen_factor)

    residual = residual_block(residual, 64*widen_factor, 2, True)
    for i in xrange(n - 1):
        residual = residual_block(residual, 64*widen_factor)

    global_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True)
    fc = L.InnerProduct(global_pool,num_output=classes,
                        bias_filler=dict(type='constant', value=0), weight_filler=dict(type='msra'))
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def make_net(tgt_file):
    with open(tgt_file, 'w') as f:
        print('name: "wide_resnet"', file=f)
        print(resnet_cifar(depth=16, widen_factor=8), file=f)

if __name__ == '__main__':
    tgt_file='wide_resnet.prototxt'
    make_net(tgt_file)
