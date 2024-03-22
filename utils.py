import torch.nn as nn
import torch
import separableconv.nn as sep
import math

   
class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None, training = False):
        w_L2 = torch.linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = torch.linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / ((x_L2 * w_L2)  + 1e-8)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = torch.nn.functional.one_hot(label, num_classes=self.cout)
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
                        
        return cos
    
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=40.0, m=0.3, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = torch.nn.Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # label = torch.nn.functional.one_hot(label, num_classes=self.out_features)
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(x), torch.nn.functional.normalize(self.weight))
        if label == None:
            return cosine
        
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            
        if len(label.shape) == 1: 
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        else:
            one_hot = label
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
    

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = torch.nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = torch.nn.functional.normalize(input)
        # normalize weights
        W = torch.nn.functional.normalize(self.W)
        # dot product
        logits = torch.nn.functional.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg + 1e-7) / (torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med)) + 1e-7)
        output = self.s * logits

        return output