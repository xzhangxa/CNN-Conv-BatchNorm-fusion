#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def get_layer_by_name(proto, name):
    for i in xrange(len(proto.layer)):
        if proto.layer[i].name == name:
            return proto.layer[i]
    return None


def get_conv_layer_name(proto, name):
    layer = get_layer_by_name(proto, name)
    if not layer:
        return None
    if layer.type == u'Scale':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
        if bottom_layer and bottom_layer.type == u'BatchNorm':
            bottom2_layer = get_layer_by_name(proto, bottom_layer.bottom[0])
            if bottom2_layer and bottom2_layer.type == u'Convolution':
                return bottom2_layer.name
    elif layer.type == u'BatchNorm':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
        if bottom_layer and bottom_layer.type == u'Convolution':
            return bottom_layer.name
    return None


class Convert:
    def __init__(self, network, model):
        self.network = network
        self.model = model
        caffe.set_mode_cpu()
        self.orig_net = caffe.Net(network, model, caffe.TEST)
        self.net = self.orig_net

    def eliminate_bn(self):
        conv_layers_bn = {}
        conv_layers_sc = {}
        proto = caffe_pb2.NetParameter()
        text_format.Merge(open(self.network).read(), proto)

        # change network topology
        for i in xrange(len(proto.layer)):
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
                continue
            for j in xrange(len(layer.top)):
                conv_layer_name = get_conv_layer_name(proto, layer.top[j])
                if conv_layer_name:
                    layer.top[j] = conv_layer_name + '/mod'
            for j in xrange(len(layer.bottom)):
                conv_layer_name = get_conv_layer_name(proto, layer.bottom[j])
                if conv_layer_name:
                    layer.bottom[j] = conv_layer_name + '/mod'

        # loop again to remove BN and SC
        i = len(proto.layer)
        while i > 0:
            i -= 1
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
                conv_layer_name = get_conv_layer_name(proto, layer.name)
                if conv_layer_name:
                    if layer.type == u'BatchNorm':
                        conv_layers_bn[conv_layer_name] = layer.name
                        conv_layer = get_layer_by_name(proto, conv_layer_name)
                        conv_layer.convolution_param.bias_term = True
                        for j in xrange(len(conv_layer.top)):
                            if conv_layer.top[j] == conv_layer.name:
                                conv_layer.top[j] += '/mod'
                        conv_layer.name += '/mod'
                    else:
                        conv_layers_sc[conv_layer_name] = layer.name
                    proto.layer.remove(layer)

        outproto = self.network.replace('.prototxt', '-m.prototxt')
        outmodel = self.model.replace('.caffemodel', '-m.caffemodel')

        with open(outproto, 'w') as f:
            f.write(str(proto))

        # calc new conv weights from original conv/bn/sc weights
        new_w = {}
        new_b = {}
        for layer in conv_layers_bn:
            old_w = self.orig_net.params[layer][0].data
            if len(self.orig_net.params[layer]) > 1:
                old_b = self.orig_net.params[layer][1].data
            else:
                old_b = np.zeros(self.orig_net.params[layer][0].data.shape[0],
                                 self.orig_net.params[layer][0].data.dtype)
            if self.orig_net.params[conv_layers_bn[layer]][2].data[0] != 0:
                s = 1 / self.orig_net.params[conv_layers_bn[layer]][2].data[0]
            else:
                s = 0
            u = self.orig_net.params[conv_layers_bn[layer]][0].data * s
            v = self.orig_net.params[conv_layers_bn[layer]][1].data * s
            alpha = self.orig_net.params[conv_layers_sc[layer]][0].data
            beta = self.orig_net.params[conv_layers_sc[layer]][1].data
            new_b[layer] = alpha * (old_b - u) / np.sqrt(v + 1e-5) + beta
            new_w[layer] = (alpha / np.sqrt(v + 1e-5))[...,
                                                       np.newaxis,
                                                       np.newaxis,
                                                       np.newaxis] * old_w

        # make new net and save model
        self.net = caffe.Net(outproto, self.model, caffe.TEST)
        for layer in new_w:
            self.net.params[layer + '/mod'][0].data[...] = new_w[layer]
            self.net.params[layer + '/mod'][1].data[...] = new_b[layer]
        self.net.save(outmodel)
        self.net = caffe.Net(outproto, outmodel, caffe.TEST)

    def test(self):
        np.random.seed()
        rand_image = np.random.rand(1, 3, 224, 224) * 255
        self.net.blobs['data'].data[...] = rand_image
        self.orig_net.blobs['data'].data[...] = rand_image

        # compute
        out = self.net.forward()
        orig_out = self.orig_net.forward()

        # predicted predicted class
        print(orig_out['prob'].argmax(), 'vs', out['prob'].argmax())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert')
    parser.add_argument('-d', '--deploy', action='store', dest='deploy',
                        required=True, help='deploy prototxt')
    parser.add_argument('-m', '--model', action='store', dest='model',
                        required=True, help='caffemodel')
    parser.add_argument('-t', '--test', action='store_true', dest='test',
                        help='run test')
    args = parser.parse_args()

    net = Convert(args.deploy, args.model)

    net.eliminate_bn()
    if args.test:
        net.test()
