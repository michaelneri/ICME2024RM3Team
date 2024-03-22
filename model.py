import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy, ConfusionMatrix
from torchinfo import summary
import torchaudio.transforms as T
from utils import ArcFace, ArcMarginProduct, AdaCos
import separableconv.nn as sep
import numpy as np
from mobilenet import MobileFaceNet

class Wavegram_AttentionModule(nn.Module):
    def __init__(self):
        super(Wavegram_AttentionModule, self).__init__()

        self.wavegram = sep.SeparableConv1d(in_channels = 1, out_channels = 128, kernel_size = 1024, stride = 512, padding = 512)

        # Mel filterbank
        self.transform_tf = T.MelSpectrogram(sample_rate=16000,
                                                n_fft=1024,
                                                win_length=1024,
                                                hop_length=512,
                                                center=True,
                                                pad_mode="reflect",
                                                power=2.0,
                                                norm="slaney",
                                                n_mels=128,
                                                mel_scale="htk",
                                                )
        # attention module
        
        self.heatmap = nn.Sequential(
                sep.SeparableConv2d(in_channels = 2, out_channels = 16, kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                sep.SeparableConv2d(in_channels = 16, out_channels = 64, kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                sep.SeparableConv2d(in_channels = 64, out_channels = 2, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            ) 

        # classifier
        self.classifier = MobileFaceNet(num_class = 10)
        self.arcface = ArcMarginProduct(in_features = 128, out_features = 10, s = 8, m = 0.2)

    def normalize_tensor(self, x):
        mean = x.mean(dim = (2,3), keepdim = True)
        std = x.std(dim = (2,3), unbiased = False, keepdim = True)
        return torch.div((x - mean), std)
    
    def forward(self, x, metadata):
        # compute mel spectrogram
        x_spec = self.transform_tf(x)
        x_spec = 10*torch.log10(x_spec + 1e-8)
        # compute wavegram
        x = x.unsqueeze(1)
        x = self.wavegram(x)
        x = torch.stack((x_spec, x), dim = 1)
        reppr = x
        x = self.normalize_tensor(x)
        heatmap = self.heatmap(x)
        x = x * heatmap
        x, features = self.classifier(x)
        x = self.arcface(features, metadata)
        return x, reppr, features, heatmap

class Wavegram_AttentionMap(LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.model = Wavegram_AttentionModule()
        self.lr = lr

        self.accuracy_training = Accuracy(task="multiclass", num_classes = 10)
        self.accuracy_val = Accuracy(task="multiclass", num_classes = 10)
        self.accuracy_test = Accuracy(task="multiclass", num_classes = 10)
        self.conf_mat = ConfusionMatrix(task = "multiclass", num_classes = 10)

    def forward(self, x, labels):
        return self.model(x, labels)
    
    def training_step(self, batch, batch_idx):
        x, labels, confidence = batch
        predicted, reppr, features, heatmap = self.forward(x, labels)
        loss = torch.nn.functional.cross_entropy(predicted, labels, reduction = "none")
        loss = torch.mean(loss * confidence)
        self.log("train/loss_class", loss, on_epoch = True, on_step = True, prog_bar = True)
        self.accuracy_training(predicted, labels)
        self.log("train/acc", self.accuracy_training, on_epoch = True, on_step = False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # confidence here should  be always 1
        x, labels, confidence = batch
        predicted, reppr, features, heatmap = self.forward(x, labels)
        loss = torch.nn.functional.cross_entropy(predicted, labels, reduction = "none")
        loss = torch.mean(loss * confidence)
        self.log("val/loss_class", loss, on_epoch = True, on_step = False, prog_bar = True)
        self.accuracy_val(predicted, labels)
        self.log("val/acc", self.accuracy_val, on_epoch = True, on_step = False)
        return loss
    
    def test_step(self, batch, batch_idx):
        # confidence here should be always 1
        x, labels, confidence = batch
        predicted, reppr, features, heatmap = self.forward(x, labels)
        loss = torch.nn.functional.cross_entropy(predicted, labels, reduction = "none")
        loss = torch.mean(loss * confidence)
        self.log("test/loss_class", loss, on_epoch = True, on_step = False, prog_bar = True)
        self.accuracy_test(predicted, labels)
        self.log("test/acc", self.accuracy_test, on_epoch = True, on_step = False)
        self.conf_mat(predicted, labels)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr = self.lr)
        # return opt
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=100, eta_min=0.1*float(self.lr))
                           },
              }   
    
    
# TEST FUNCTION
if __name__ == "__main__":
    example_input = torch.rand(16, 160000)
    model = Wavegram_AttentionModule()
    metadata = torch.nn.functional.one_hot(torch.randint(low = 0, high = 10, size =(16,)), num_classes=10)
    output = model(example_input, metadata)
    print(output)
    summary(model, input_data = [example_input, metadata])
    
