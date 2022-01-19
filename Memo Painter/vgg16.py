import torch.nn as tnn
import torch
import torch.nn.functional as F



def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        AdaptiveInstanceNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        AdaptiveInstanceNorm2d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, input_nc, output_nc, ngf, color_dim = 313):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([input_nc,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [2,2,2], 2, 2)
        self.layer7 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [2,2,2], 2, 2)

        # Final layer
        self.layer8 = tnn.Linear(4096, output_nc)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7,
                       self.layer8]
        self.tan=tnn.Tanh()
        self.mlp = MLP(color_dim, self.get_num_adain_params(self.layers), self.get_num_adain_params(self.layers), 3)

        self.relu = tnn.ReLU(inplace=True)
        self.deconv1 = tnn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 =  tnn.BatchNorm2d(512)
        self.deconv2 = tnn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = tnn.BatchNorm2d(256)
        self.deconv3 = tnn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = tnn.BatchNorm2d(128)
        self.deconv4 = tnn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = tnn.BatchNorm2d(64)
        self.deconv5 = tnn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = tnn.BatchNorm2d(32)
        self.classifier = tnn.Conv2d(32, output_nc, kernel_size=1)
    def forward(self, x,color_feat):
        ### AdaIn params
        adain_params = self.mlp(color_feat)
        self.assign_adain_params(adain_params, self.layers)

        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer6(out)
        # print(out.shape)
        out = self.layer7(out)
        # print(out.shape)
        # out = self.layer8(out)
        # print(out.shape)
        score = self.bn1(self.relu(self.deconv1(out)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)

        return score

    def get_num_adain_params(self, _module):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params, _module):
        # assign the adain_params to the AdaIN layers in model
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2 * m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2 * m.num_features:
                        adain_params = adain_params[:, 2 * m.num_features:]


class MLP(tnn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, act=tnn.ReLU(inplace=True)):
        super(MLP, self).__init__()
        self.model = []

        self.model.append(tnn.Linear(input_dim, dim))
        self.model.append(act)

        for i in range(n_blk - 2):
            self.model.append(tnn.Linear(dim, dim))
            self.model.append(act)

        self.model.append(tnn.Linear(dim, output_dim))
        self.model = tnn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class AdaptiveInstanceNorm2d(tnn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
