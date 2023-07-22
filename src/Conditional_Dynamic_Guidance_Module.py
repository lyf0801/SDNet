import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.pool = nn.AdaptiveAvgPool2d(output_size=(28, 28))
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.pool(x)
        b, c, h, w = x.size()
        att = x.reshape(b, h*w, 1, 1) #bs,dim,1,1
        att = self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)

"""
Conditional Dynamic Guidance Module
"""
class CDGM(nn.Module):

    def __init__(self,channels, kernel_size=3,grounps=1, K=16,temprature=30,ratio=4):
        super(CDGM, self).__init__()
        self.channels=channels
        self.K=K
        self.dim = 28*28
        self.kernel_size = 3 
        self.attention=Attention(in_planes=self.dim, ratio=ratio,K=K,temprature=temprature)
        self.weight=nn.Parameter(torch.randn(K,self.channels,self.channels//grounps,kernel_size,kernel_size),requires_grad=True)
        self.bias=nn.Parameter(torch.randn(K,self.channels),requires_grad=True)
    
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x, scene_knowledge):
        input = x
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(scene_knowledge) #bs,K
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.channels,self.channels,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k

        bias=self.bias.view(self.K,-1) #K,out_p
        aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
        output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=1,padding=1,groups=bs)
      
        #residual learning
        output = output.view(bs,self.channels,h,w) + input

        return output

if __name__ == '__main__':
    x=torch.randn(2,256,224,224)
    scene = torch.randn(2,1,224,224)
    m=CDGM(channels=256, dim=224)
    out=m(x, scene)
    print(out.shape)