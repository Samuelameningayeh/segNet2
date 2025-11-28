# model_segnet.py
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SegNet(nn.Module):
    """
    Full SegNet: VGG16-like encoder (13 conv) + symmetric decoder
    with max-pooling indices for upsampling.
    """
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # ----- Encoder -----
        # Block 1
        self.enc1_1 = ConvBNReLU(in_channels, 64)
        self.enc1_2 = ConvBNReLU(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 2
        self.enc2_1 = ConvBNReLU(64, 128)
        self.enc2_2 = ConvBNReLU(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 3
        self.enc3_1 = ConvBNReLU(128, 256)
        self.enc3_2 = ConvBNReLU(256, 256)
        self.enc3_3 = ConvBNReLU(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 4
        self.enc4_1 = ConvBNReLU(256, 512)
        self.enc4_2 = ConvBNReLU(512, 512)
        self.enc4_3 = ConvBNReLU(512, 512)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 5
        self.enc5_1 = ConvBNReLU(512, 512)
        self.enc5_2 = ConvBNReLU(512, 512)
        self.enc5_3 = ConvBNReLU(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        # ----- Decoder -----
        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.dec5_1 = ConvBNReLU(512, 512)
        self.dec5_2 = ConvBNReLU(512, 512)
        self.dec5_3 = ConvBNReLU(512, 512)

        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4_1 = ConvBNReLU(512, 512)
        self.dec4_2 = ConvBNReLU(512, 512)
        self.dec4_3 = ConvBNReLU(512, 256)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3_1 = ConvBNReLU(256, 256)
        self.dec3_2 = ConvBNReLU(256, 256)
        self.dec3_3 = ConvBNReLU(256, 128)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2_1 = ConvBNReLU(128, 128)
        self.dec2_2 = ConvBNReLU(128, 64)

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1_1 = ConvBNReLU(64, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x = self.enc1_1(x)
        x = self.enc1_2(x)
        size1 = x.size()
        x, idx1 = self.pool1(x)

        x = self.enc2_1(x)
        x = self.enc2_2(x)
        size2 = x.size()
        x, idx2 = self.pool2(x)

        x = self.enc3_1(x)
        x = self.enc3_2(x)
        x = self.enc3_3(x)
        size3 = x.size()
        x, idx3 = self.pool3(x)

        x = self.enc4_1(x)
        x = self.enc4_2(x)
        x = self.enc4_3(x)
        size4 = x.size()
        x, idx4 = self.pool4(x)

        x = self.enc5_1(x)
        x = self.enc5_2(x)
        x = self.enc5_3(x)
        size5 = x.size()
        x, idx5 = self.pool5(x)

        # ----- Decoder -----
        x = self.unpool5(x, idx5, output_size=size5)
        x = self.dec5_1(x)
        x = self.dec5_2(x)
        x = self.dec5_3(x)

        x = self.unpool4(x, idx4, output_size=size4)
        x = self.dec4_1(x)
        x = self.dec4_2(x)
        x = self.dec4_3(x)

        x = self.unpool3(x, idx3, output_size=size3)
        x = self.dec3_1(x)
        x = self.dec3_2(x)
        x = self.dec3_3(x)

        x = self.unpool2(x, idx2, output_size=size2)
        x = self.dec2_1(x)
        x = self.dec2_2(x)

        x = self.unpool1(x, idx1, output_size=size1)
        x = self.dec1_1(x)

        x = self.classifier(x)
        return x
