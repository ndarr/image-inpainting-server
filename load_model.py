# -*- coding: utf-8 -*-
"""loadModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nU4CGakpgLLPdCljEBHRnASXtmaRYB4Q
"""


# Parts of Autoencoder Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d
from torch.nn import Sequential, ReLU, BatchNorm2d, ReflectionPad2d, Conv2d, ConvTranspose2d, MaxPool2d, MaxUnpool2d
import torch

class LeakyTanh(torch.nn.Module):
  def __init__(self, leak_factor: float = 0.01, **kwargs):
    """
    Tanh activation function with LeakyReLU-like slope.
    Args:
      leak_factor: added slope of the function
    """
    super(LeakyTanh, self).__init__(**kwargs)
    self.leak_factor = leak_factor

  def forward(self, inputs, **kwargs):
    return torch.add(torch.tanh(inputs), torch.mul(inputs, self.leak_factor))

class DoubleConv(nn.Module):
  """(partialConvolution => [BN] => LeakyTanh or LeakyReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
      PartialConv2d(in_channels, mid_channels, 3, multi_channel=True, padding = 1),
      nn.BatchNorm2d(mid_channels),
      LeakyTanh(),
      PartialConv2d(mid_channels, out_channels, 3, multi_channel=True, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='nearest')
      self.conv = DoubleConv(in_channels, out_channels)
    else:
      self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

class UNetInspired(torch.nn.Module):
  def __init__(self, bilinear=True):
    super(UNetInspired, self).__init__()
    
    layer1Depth = 64
    layer2Depth = layer1Depth*2
    layer3Depth = layer2Depth*2
    layer4Depth = layer3Depth*2

    self.inc = DoubleConv(4, layer1Depth)
    self.down1 = Down(layer1Depth, layer2Depth)
    self.down2 = Down(layer2Depth, layer3Depth)
    self.down3 = Down(layer3Depth, layer4Depth)
    self.up1 = Up(layer4Depth + layer3Depth, layer3Depth, bilinear=True)
    self.up2 = Up(layer3Depth + layer2Depth, layer2Depth, bilinear=True)
    self.up3 = Up(layer2Depth + layer1Depth, layer1Depth, bilinear=True)
    self.outc = OutConv(layer1Depth, 3)

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x = self.up1(x4, x3)
    x = self.up2(x, x2)
    x = self.up3(x, x1)
    out = self.outc(x)
    return out

def load_model(filename):
  model = UNetInspired()
  model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

  device = torch.device("cpu")
  model.to(device)
  model.eval()
  return model