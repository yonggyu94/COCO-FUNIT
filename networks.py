"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, AdainUpResBlock


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class GPPatchMcResDis(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(3, nf, 3, 1, 1,
                             pad_type='reflect',
                             norm='none',
                             activation='none',
                             sn=True)]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none', sn=True)]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none', sn=True)]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none', sn=True)]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none', sn=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
                                 norm='none',
                                 activation='lrelu',
                                 activation_first=True,
                                 sn=True)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)            # {B, 1024, 4, 4}
        out = self.cnn_c(feat)          # [B, C, 4, 4]
        index = torch.LongTensor(range(out.size(0))).cuda()     # [64] 0 ~ 64 Select batch
        out1 = out[index, y, :, :]      # [B, 4, 4] batch 마다 y번째 patch patch를 선택
        return out1, feat

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)     # [B, 4, 4], [B, 1024, 4, 4]
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()            # tensor[1024]
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()     # tensor[], -1보다 큰 값에 loss
        correct_count = (resp_fake < 0).sum()                   # B*4*4 중에서 < 0 의 수를 c_c
        fake_accuracy = correct_count.type_as(fake_loss) / total_count  # c_c / B*4*4(t_c)
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)     # [B, 4, 4], [B, 1024, 4, 4]
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()            # tensor[1024]
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()     # tensor[], 1보다 작은 값에 loss
        correct_count = (resp_real >= 0).sum()                  # B*4*4 중에서 >= 0 의 수를 c_c
        real_accuracy = correct_count.type_as(real_loss) / total_count  # c_c / B*4*4(t_c)
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)       # [B, C, H, W]
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg


class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        down = (down_class, down_content)
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']
        batch_size = hp['batch_size']
        self.enc_class_model = ClassModelEncoder(batch_size,
                                                 down,
                                                 3,
                                                 nf,
                                                 latent_dim,
                                                 norm='none',
                                                 activ='relu',
                                                 pad_type='reflect')

        self.enc_content = ContentEncoder(down_content,
                                          n_res_blks,
                                          3,
                                          nf,
                                          'in',
                                          activ='relu',
                                          pad_type='reflect')

        self.dec = Decoder(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

        self.mlp = MLP(latent_dim,
                       get_num_adain_params(self.dec),
                       nf_mlp,
                       n_mlp_blks,
                       norm='none',
                       activ='relu')

    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(one_image, model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, model_code):
        # decode content and style codes to an image
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images


class ClassModelEncoder(nn.Module):
    def __init__(self, batch_size, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        s_s_layers = []
        dim_size = dim
        for i in range(downs[0]):
            if i == 0:
                s_s_layers.append(Conv2dBlock(ind_im, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            else:
                dim = dim*2
                s_s_layers.append(Conv2dBlock(dim // 2, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        self.enc_s_s = nn.Sequential(*s_s_layers)

        dim = dim_size
        s_c_layers = []
        for i in range(downs[1]):
            if i == 0:
                s_c_layers.append(Conv2dBlock(ind_im, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            else:
                dim = dim*2
                s_c_layers.append(Conv2dBlock(dim // 2, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        dim = dim * 2
        s_c_layers.append(Conv2dBlock(dim // 2, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type))
        s_c_layers.append(ResBlocks(2, dim, norm=norm, activation=activ, pad_type=pad_type))
        self.enc_s_c = nn.Sequential(*s_c_layers)

        self.linear_s = nn.Linear(dim*2, latent_dim)
        self.linear_c = nn.Linear(dim, latent_dim)

        self.csb = torch.randn(batch_size, dim).cuda()

    def forward(self, c_img, s_img):
        c_out = self.enc_s_c(c_img)
        s_out = self.enc_s_s(s_img)

        c_out = torch.mean(c_out, dim=[2, 3])  # [B, 512]
        s_out = torch.mean(s_out, dim=[2, 3])  # [B, 512]
        s_out = torch.cat([s_out, self.csb], dim=1)

        c_out = self.linear_c(c_out)
        s_out = self.linear_s(s_out)

        out = c_out * s_out
        return out


class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()

        s_c_layers = []
        for i in range(downs):
            if i == 0:
                s_c_layers.append(Conv2dBlock(input_dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            else:
                dim = dim * 2
                s_c_layers.append(Conv2dBlock(dim // 2, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        dim = dim * 2
        s_c_layers.append(Conv2dBlock(dim // 2, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type))
        s_c_layers.append(ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type))
        self.model = nn.Sequential(*s_c_layers)
        self.output_dim = dim

    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(ups):
            self.model.append(AdainUpResBlock(dim, activation=activ, pad_type=pad_type))
            dim = dim // 2
        self.model.append(Conv2dBlock(dim, out_dim, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
