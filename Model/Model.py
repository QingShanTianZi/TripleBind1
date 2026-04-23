import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class MBCN1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv3 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)

        self.se = SEBlock(out_ch * 3)

    def forward(self, x):

        x_middle = x
        x_middle = x_middle.squeeze(-1)

        esm2 = x_middle[:, :1280]
        prott5 = x_middle[:, 1280:1280 + 1024]
        ankh = x_middle[:, 1280 + 1024:]

        esm2 = F.pad(esm2, (0, 1536 - 1280))
        prott5 = F.pad(prott5, (0, 1536 - 1024))

        out = torch.stack([esm2, prott5, ankh], dim=2)

        x_cat = torch.cat([
            self.conv1(out),
            self.conv2(out),
            self.conv3(out)
        ], dim=1)

        return self.se(x_cat)


class MBCN2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = CNN1D(in_ch, out_ch, kernel_size=3)
        self.conv2 = CNN1D(in_ch, out_ch, kernel_size=3)
        self.conv3 = CNN1D(in_ch, out_ch, kernel_size=3)

        self.se = SEBlock(out_ch * 3)

    def forward(self, x):

        x_cat = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x)
        ], dim=1)

        return self.se(x_cat)


class MBCN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv3 = CNN1D(in_ch, out_ch, kernel_size=3, padding=1)

        self.se = SEBlock(out_ch * 3)

    def forward(self, x):

        x_cat = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x)
        ], dim=1)

        return self.se(x_cat)



class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = MBCN1(1536, 128)
        self.block2 = MBCN2(384, 64)
        self.block3 = MBCN(192, 64)
        self.block4 = MBCN(192, 32)
        self.mlp = nn.Sequential(
            nn.Linear(96, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.block4(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.mlp(x)
        return x.squeeze(1)