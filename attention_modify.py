
import torch
import torch.nn as nn
import pdb
import torchvision.models as models
import torch.nn.functional as F

def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.mul(size)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div

class LocalResponseNorm(nn.Module):
    r"""Applies local response normalization over an input signal composed
    of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, ...)`
        - Output: :math:`(N, C, ...)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)

class Layer_SE(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(Layer_SE, self).__init__()
        self.chanel_in = in_dim
        self.fc1 = nn.Linear(in_features=4, out_features=4)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        m_batchsize, N, C, height, width = x.size()
        original_out = x
        out = x.view(m_batchsize, N, -1).mean(2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1, 1)
        out = out * original_out
        return out

class Resnet18_attention_new(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18_attention_new, self).__init__()
        # Original Model
        self.backbone_head = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.backbone = models.resnet18(pretrained)
        
        # Universal Module
        self.maxpool0 = nn.MaxPool2d(kernel_size=8, stride=8) # W * H --> W/8 * H/8
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=8) # W * H --> W/8 * H/8
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4) # W * H --> W/4 * H/4
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # W * H --> W/2 * H/2

        self.conv0 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv1 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv4 = nn.Conv2d(2560, 512, kernel_size=3, stride=1, padding=1,bias=False)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))                # B * C * W * H --> B * C * 1 * 1
        # Expression Recognition
        self.LRN_em = LocalResponseNorm(2)
        self.reduce_dim_em = nn.Sequential(
            nn.Conv2d(in_channels=957,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        ) 
        self.pred_ex = nn.Linear(in_features=512, out_features=7, bias=True)
        #layer attention module
        self.lam = Layer_SE(512)

    def forward(self, input):

        featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone_head(input)))) # 64 * 56 * 56
        
        featureMap1 = self.backbone.layer1(featureMap0)                        # 64 * 56 * 56
        featureMap2 = self.backbone.layer2(featureMap1)                        # 128 * 28 * 28
        featureMap3 = self.backbone.layer3(featureMap2)                        # 256 * 14 * 14
        featureMap4 = self.backbone.layer4(featureMap3)                        # 512 * 7 * 7

        featureMap0 = self.maxpool0(featureMap0)                               # 64 * 56 * 56 --> 64 * 7 * 7
        
        featureMap1 = self.maxpool1(featureMap1)                               # 64 * 56 * 56 --> 64 * 7 * 7
        featureMap2 = self.maxpool2(featureMap2)                               # 128 * 28 * 28 --> 128 * 7 * 7
        featureMap3 = self.maxpool3(featureMap3)                               # 256 * 14 * 14 --> 256 * 7 * 7

        featureMap0 = self.conv1(featureMap0)                               # 64 * 7 * 7 --> 512 * 7 * 7
        featureMap1 = self.conv1(featureMap1)                               # 64 * 7 * 7 --> 512 * 7 * 7
        featureMap2 = self.conv2(featureMap2)                               # 128 * 7 * 7 --> 512 * 7 * 7
        featureMap3 = self.conv3(featureMap3)                               # 256 * 7 * 7 --> 512 * 7 * 7
        
        featureMap = torch.cat((torch.cat((featureMap1.unsqueeze(1), featureMap2.unsqueeze(1)), dim=1), torch.cat((featureMap3.unsqueeze(1), featureMap4.unsqueeze(1)), dim=1)), dim=1) # 4 * 512 * 7 * 7

        # Save GPU Memory
        # del featureMap0, featureMap1, featureMap2, featureMap3, featureMap4
        # torch.cuda.empty_cache()

        # featureMap = self.LRN_em(featureMap)                                   # 4 * 512 * 7 * 7 --> 4 * 512 * 7 * 7 
        
        featureMap5 =  self.lam(featureMap)     # 4 * 512 * 7 * 7 --> 4 * 512 * 7 * 7 
        featureMap5 = featureMap5.view(featureMap5.size(0),-1,featureMap5.size(3),featureMap5.size(4))  # 4 * 512 * 7 * 7 --> 2048 * 7 * 7 

        featureMap6 = torch.cat((featureMap5,featureMap4),dim=1) # 2048 * 7 * 7 --> 2560 * 7 * 7
        featureMap6 = self.conv4(featureMap6)
        
        featureMap6 += featureMap0 # 1024 * 7 * 7 --> 1024 * 7 * 7

        feature = self.GAP(featureMap6)     # 1024 * 7 * 7 --> 1024 * 1 * 1
        feature = feature.view(feature.size(0),feature.size(1))                # 1024 * 1 * 1 --> 1024
        pred = self.pred_ex(feature)                                           # 1024 --> 7

        return pred



