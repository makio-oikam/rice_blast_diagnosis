# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
# 学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import resnet18

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self, finetune_layers=5): # 層数の指定
        super().__init__()

        self.feature = resnet18(pretrained=True)

        # 後ろから数えて指定された層数分をファインチューニング
        all_params = list(self.feature.parameters())
        for param in all_params[:-finetune_layers]:
            param.requires_grad = False

        self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
    
    