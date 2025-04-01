from config import nn,np


class SimplifiedVGG(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(SimplifiedVGG, self).__init__()
        self.features = nn.Sequential(
    
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )


        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, num_classes)
)



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

