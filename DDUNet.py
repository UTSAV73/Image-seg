import torch
import torch.nn as nn
import torch.nn.functional as F

#implementation of Dilated Dense Unet in pytorch according to dals model

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.01
regularizer = 0.0008
dropout_rate = 0.3
growth_k = 6

class BatchNormalization(nn.Module):
    def __init__(self, num_features):
        super(BatchNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=_BATCH_NORM_EPSILON, momentum=1 - _BATCH_NORM_DECAY)

    def forward(self, x):
        return self.batch_norm(x)


def conv_norm_relu(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        BatchNormalization(out_channels),
        nn.ReLU(inplace=True)
    )

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, growth_k):
        super(TransitionLayer, self).__init__()
        self.bn = BatchNormalization(in_channels)
        self.conv = conv_norm_relu(in_channels, growth_k, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_k, dilation, is_training):
        super(BottleneckLayer, self).__init__()
        self.is_training = is_training
        self.bn1 = BatchNormalization(in_channels)
        self.conv1 = conv_norm_relu(in_channels, 4 * growth_k, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.bn2 = BatchNormalization(4 * growth_k)
        self.conv2 = conv_norm_relu(4 * growth_k, 4 * growth_k, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, nb_layers, growth_k, dilation, is_training):
        super(DenseBlock, self).__init__()
        layers = [BottleneckLayer(in_channels, growth_k, dilation, is_training)]
        for _ in range(nb_layers - 1):
            layers.append(BottleneckLayer(in_channels + (growth_k * 4), growth_k, dilation, is_training))
            in_channels += growth_k * 4
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class DDUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DDUNet, self).__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.final_conv(up_4)
        return out

#debugging
# if __name__ =="__main__":
#     double_conv=DoubleConv(256,256)
#     print(double_conv)
#     input_image=torch.rand((1,3,256,256))
#     model=UNet(3,10)
#     output=model(input_image)
#     print(output.size())