
import torch
from torch import nn
from torch.nn import functional as F

class ODAM_Loss(nn.Module):
    def __init__(self):
        super(ODAM_Loss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        boundary_targets = F.conv2d(gtmasks, self.laplacian_kernel, padding=1)
        boundary_targets_x2 = F.conv2d(gtmasks, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        boundary_targets_x4 = F.conv2d(gtmasks, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
       

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)
        """
        0-1 mask
        """
        boudary_targets_pyramid[boudary_targets_pyramid > 0.5] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.5] = 0

       

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
   
        #mean
        return bce_loss.mean()

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params

"""
Object Detail Awareness Module
"""
class ODAM(nn.Module):
    def __init__(self, in_channel = 256, mid_channel=64):
        super(ODAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, 1, kernel_size=1, bias=False)
        )
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
