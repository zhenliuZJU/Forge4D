import torch
import torch.nn as nn

class GaussianFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GaussianFusion, self).__init__()
        
        # 第一层 1x1 卷积 + ReLU
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
        # 第二层 1x1 卷积
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # nn.init.zeros_(self.conv2.weight)
        # if self.conv2.bias is not None:
        #     nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x):
        # 第一层 1x1 卷积 + ReLU
        out = self.conv1(x)
        out = self.relu(out)
        
        # 第二层 1x1 卷积（无残差，直接输出）
        out = self.conv2(out)
        
        return out

# 测试
if __name__ == "__main__":
    # 情况1：输入输出通道相同
    x_same = torch.randn(1, 64, 32, 32)  # (B, C, H, W)
    model_same = GaussianFusion(64, 32, 64)
    out_same = model_same(x_same)
    print("输入输出通道相同（无残差）:")
    print("输入形状:", x_same.shape)
    print("输出形状:", out_same.shape)
    
    # 情况2：输入输出通道不同
    x_diff = torch.randn(1, 64, 32, 32)
    model_diff = Simple1x1CNN(64, 32, 128)
    out_diff = model_diff(x_diff)
    print("\n输入输出通道不同（无残差）:")
    print("输入形状:", x_diff.shape)
    print("输出形状:", out_diff.shape)