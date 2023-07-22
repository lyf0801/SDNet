
#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline import ResNet,PPM,FuseBlock,SOD_Head,weight_init
from src.resnet18_cam import resnet18_cam
from src.Scene_Knowledge_Transfer_Module import PoolFormerBlock
from src.Conditional_Dynamic_Guidance_Module import CDGM
from src.Object_Detail_Awareness_Module import ODAM

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.bkbone  = ResNet()
        self.ppm = PPM(down_dim=256)

        self.fuse5 = FuseBlock(in_channel1 = 2048,  in_channel2 = 256)
        self.fuse4 = FuseBlock(in_channel1 = 1024,  in_channel2 = 256)
        self.fuse3 = FuseBlock(in_channel1 = 512,  in_channel2 = 256)
        self.fuse2 = FuseBlock(in_channel1 = 256,  in_channel2 = 256)
        self.fuse1 = FuseBlock(in_channel1 = 64,  in_channel2 = 256)

        self.scene_prediction = resnet18_cam(pretrained=True, num_classes=12) 

        self.PoolFusion1 = PoolFormerBlock(dim = 4)
        self.PoolFusion2 = PoolFormerBlock(dim = 4)
        self.PoolFusion3 = PoolFormerBlock(dim = 4)
        self.PoolFusion4 = PoolFormerBlock(dim = 4)
        self.PoolFusion5 = PoolFormerBlock(dim = 4)

        self.CDGM1 = CDGM(channels=256)
        self.CDGM2 = CDGM(channels=256)
        self.CDGM3 = CDGM(channels=256)
        self.CDGM4 = CDGM(channels=256)
        self.CDGM5 = CDGM(channels=256)

        self.SOD_head1 = SOD_Head()
        self.SOD_head2 = SOD_Head()
        self.SOD_head3 = SOD_Head()
        self.SOD_head4 = SOD_Head()
        self.SOD_head5 = SOD_Head()

        self.ODAM1 = ODAM(in_channel=64)
        self.ODAM2 = ODAM(in_channel=256)
        self.ODAM3 = ODAM(in_channel=512)
        """
        The architecture is a typical classification network with global average pooling (GAP) followed by a
        fully connected layer, and is trained by a classification criteria with image-level labels
        """
        self.initialize()

    def forward(self, x):
        """
        baseline operations
        """
        s1,s2,s3,s4,s5 = self.bkbone(x)
        s6 = self.ppm(s5)

        out5 =  self.fuse5(s5, s6)

        out4 =  self.fuse4(s4, F.interpolate(out5, size = s4.size()[2:], mode='bilinear',align_corners=True))

        out3  = self.fuse3(s3, F.interpolate(out4, size = s3.size()[2:], mode='bilinear',align_corners=True))

        out2  = self.fuse2(s2, F.interpolate(out3, size = s2.size()[2:], mode='bilinear',align_corners=True))

        out1  = self.fuse1(s1, F.interpolate(out2, size = s1.size()[2:], mode='bilinear',align_corners=True))
        """
        Step1: Scene Knowledge Transfer Module
        """
        x_224 = F.interpolate(x, size = (224, 224), mode='bilinear',align_corners=True)
        logits, cams, scene_feature = self.scene_prediction(x_224)
        #print("before:", (cams * scene_feature).size())
        scene_feature[0] = F.interpolate(scene_feature[0], size = out1.size()[2:], mode='bilinear',align_corners=True)
        scene_feature[1] = F.interpolate(scene_feature[1], size = out2.size()[2:], mode='bilinear',align_corners=True)
        scene_feature[2] = F.interpolate(scene_feature[2], size = out3.size()[2:], mode='bilinear',align_corners=True)
        scene_feature[3] = F.interpolate(scene_feature[3], size = out4.size()[2:], mode='bilinear',align_corners=True)
        scene_feature[4] = F.interpolate(scene_feature[4], size = out5.size()[2:], mode='bilinear',align_corners=True)
        #using metaformer to explore pixel-relation and generate scene context vector
        scene_knowledge1 = self.PoolFusion1(F.interpolate(F.interpolate(cams, size = out1.size()[2:],mode='bilinear',align_corners=True) * torch.mean(out1 * scene_feature[0], dim = 1, keepdim = True), size = out1.size()[2:], mode='bilinear',align_corners=True))
        scene_knowledge2 = self.PoolFusion2(F.interpolate(F.interpolate(cams, size = out2.size()[2:],mode='bilinear',align_corners=True) * torch.mean(out2 * scene_feature[1], dim = 1, keepdim = True), size = out2.size()[2:], mode='bilinear',align_corners=True))
        scene_knowledge3 = self.PoolFusion3(F.interpolate(F.interpolate(cams, size = out3.size()[2:],mode='bilinear',align_corners=True) * torch.mean(out3 * scene_feature[2], dim = 1, keepdim = True), size = out3.size()[2:], mode='bilinear',align_corners=True))
        scene_knowledge4 = self.PoolFusion4(F.interpolate(F.interpolate(cams, size = out4.size()[2:],mode='bilinear',align_corners=True) * torch.mean(out4 * scene_feature[3], dim = 1, keepdim = True), size = out4.size()[2:], mode='bilinear',align_corners=True))
        scene_knowledge5 = self.PoolFusion5(F.interpolate(F.interpolate(cams, size = out5.size()[2:],mode='bilinear',align_corners=True) * torch.mean(out5 * scene_feature[4], dim = 1, keepdim = True), size = out5.size()[2:], mode='bilinear',align_corners=True))
        
        
        """
        Step2: Conditional Dynamic Guidance Module
        scene_knowledge1-5 & out1-out5
        [B, 1, H, W]  [B, 256, H, W]
        """
        final_out1 = self.CDGM1(out1, scene_knowledge1)
        final_out2 = self.CDGM2(out2, scene_knowledge2)
        final_out3 = self.CDGM3(out3, scene_knowledge3)
        final_out4 = self.CDGM4(out4, scene_knowledge4)
        final_out5 = self.CDGM5(out5, scene_knowledge5)
        
        """
        Step3: Object Detail Awareness Module 
        """
        detail1 = self.ODAM1(s1)
        detail2 = self.ODAM2(s2)
        detail3 = self.ODAM3(s3)
        """
        SOD 
        """
        smap1 = self.SOD_head1(final_out1 + F.interpolate(final_out2, size = final_out1.size()[2:], mode='bilinear',align_corners=True))
        smap2 = self.SOD_head2(final_out2 + F.interpolate(final_out3, size = final_out2.size()[2:], mode='bilinear',align_corners=True))
        smap3 = self.SOD_head3(final_out3 + F.interpolate(final_out4, size = final_out3.size()[2:], mode='bilinear',align_corners=True))
        smap4 = self.SOD_head4(final_out4 + F.interpolate(final_out5, size = final_out4.size()[2:], mode='bilinear',align_corners=True))
        smap5 = self.SOD_head5(final_out5)
        ### interpolate
        smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
        smap2 = F.interpolate(smap2, size = x.size()[2:], mode='bilinear',align_corners=True)
        smap3 = F.interpolate(smap3, size = x.size()[2:], mode='bilinear',align_corners=True)
        smap4 = F.interpolate(smap4, size = x.size()[2:], mode='bilinear',align_corners=True)
        smap5 = F.interpolate(smap5, size = x.size()[2:], mode='bilinear',align_corners=True)

        if self.training:
            return smap1, smap2, smap3, smap4, smap5, logits, detail1, detail2, detail3
        else: #inference 
            return torch.sigmoid(smap1)


    def initialize(self):
        weight_init(self)
