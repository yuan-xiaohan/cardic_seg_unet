import torch
import torch.nn as nn

class Double_conv(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_down(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x

class Conv_up(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1

class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 32)
        self.Conv_down2 = Conv_down(32, 64)
        self.Conv_down3 = Conv_down(64, 128)
        self.Conv_down4 = Conv_down(128, 256)
        self.Conv_down5 = Conv_down(256, 512)
        self.Conv_up1 = Conv_up(512, 256)
        self.Conv_up2 = Conv_up(256, 128)
        self.Conv_up3 = Conv_up(128, 64)
        self.Conv_up4 = Conv_up(64, 32)
        self.Conv_out = nn.Conv2d(32, out_channels, 1, padding=0, stride=1)
        #self.Conv_final = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)

    def forward(self, x):

        x, conv1 = self.Conv_down1(x)
        #print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        #print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        #print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        #print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        #print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        #print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        #print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        #print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x, conv1)
        x = self.Conv_out(x)
        #x = self.Conv_final(x)

        return x

if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 256, 256)
    print(im.shape)
    model = U_Net(1, 2)
    x = model(im)
    print(model)
    print(x.shape)
    del model
    del x
    #print(x.shape)