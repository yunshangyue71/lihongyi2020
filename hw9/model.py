import torch.nn as nn
#
# class AE(nn.Module):
#     def __init__(self):
#         super(AE, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, 3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 5, stride=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 9, stride=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, 17, stride=1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x = self.decoder(x1)
#         return x1, x
#
# class AE(nn.Module):
#     def __init__(self):
#         super(AE, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(2)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 5, stride=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 9, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, 17, stride=1),
#             # nn.BatchNorm2d(3),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x = self.decoder(x1)
#         return x1, x
#
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),


            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),




            nn.ConvTranspose2d(64, 3, 17, stride=1),

            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x


#
# class AE(nn.Module):
#     def __init__(self):
#         super(AE, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 256, 2, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             nn.MaxPool2d(2)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, stride=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             #
#             nn.ConvTranspose2d(128, 64, 5, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             #
#             nn.ConvTranspose2d(64, 32, 9, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(32, 3, 18, stride=1),
#             # nn.BatchNorm2d(3),
#             # nn.ReLU(True),
#             #
#             # nn.ConvTranspose2d(16,3 , 8, stride=1),
#             # nn.BatchNorm2d(3),
#             # nn.ReLU(True)
#
#             # nn.ConvTranspose2d(128, 64, 9, stride=1),
#             # nn.BatchNorm2d(64),
#             # nn.ReLU(True),
#             #
#             #
#             #
#             #
#             # nn.ConvTranspose2d(64, 3, 17, stride=1),
#
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x = self.decoder(x1)
#         return x1, x

