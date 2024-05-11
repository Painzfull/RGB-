import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time 
import torch.utils.data
import cv2
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff
#%%
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, photo_paths, mask_paths, transform=None, mask_transform=None, device=torch.device('cuda')):
        self.photo_paths = photo_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        self.device = device

    def __getitem__(self, idx):
        photo = Image.open(self.photo_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            photo = self.transform(photo).to(self.device)
        if self.mask_transform:
            mask = self.mask_transform(mask).to(self.device)

        return photo, mask

    def __len__(self):
        return len(self.photo_paths)

def load_data(photo_dir, mask_dir, test_size=0.3):
    photo_files = [os.path.join(photo_dir, img) for img in os.listdir(photo_dir)]
    mask_files = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir)]

    photo_files.sort()
    mask_files.sort()

    # Create train and test datasets while preserving pairs
    train_photo_paths, test_photo_paths, train_mask_paths, test_mask_paths = train_test_split(
        photo_files, mask_files, test_size=test_size, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Tek kanallı (grayscale) maskeyi 3 kanala çoğaltır
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset(train_photo_paths, train_mask_paths, transform=transform, mask_transform=mask_transform, device=device)
    test_dataset = CustomDataset(test_photo_paths, test_mask_paths, transform=transform, mask_transform=mask_transform, device=device)

    print("Training Dataset Sample Count:", len(train_dataset))
    print("Test Dataset Sample Count:", len(test_dataset))

    return train_dataset, test_dataset

photo_dir = "C:/Users/Froggremann/Desktop/data/imgs"
mask_dir = "C:/Users/Froggremann/Desktop/data/masks"

train_dataset, test_dataset = load_data(photo_dir, mask_dir, test_size=0.3)

batch_size = 1
learning_rate = 0.0001
num_epochs = 100
num_classes = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, (photo, mask) in enumerate(train_loader):
    print(f"Photo Tensor Shape: {photo.shape}")
    print(f"Mask Tensor Shape: {mask.shape}")

    if i == 0:
        example_photo_pixels = photo[0].squeeze().cpu().numpy()
        example_mask_pixels = mask[0].squeeze().cpu().numpy()
        
        print("Example Photo pixel values:\n", example_photo_pixels)
        print("Example Mask pixel values:\n", example_mask_pixels)
        break

#%%
import matplotlib.pyplot as plt
import torch

def plot_image_mask_pair(photo, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Photo
    axes[0].imshow(photo.permute(1, 2, 0).cpu().numpy())  # Squeeze ve cmap eklemiyoruz
    axes[0].set_title('Photo')
    axes[0].axis('off')

    # Mask
    axes[1].imshow(mask.permute(1, 2, 0).squeeze().cpu().numpy(), cmap='gray')  # Mask üzerinde squeeze ve cmap
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Örnek olarak ilk fotoğraf ve maskeyi çizdirelim
example_photo, example_mask = train_dataset[1]  # Örnek olarak train setinden ilk öğeyi alıyoruz
plot_image_mask_pair(example_photo, example_mask)
#%%
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        
        out = self.out(conv9)
        return out

# Cihaz seçimi (GPU öncelikli, CPU kullanımı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Çıkış kanal sayısı
num_classes = 3

# Model oluşturma (RGB çıkış için 3 kanal)
model = UNet(in_channels=3, out_channels=num_classes)

# Modeli seçilen cihaza gönderme
model = model.to(device)

# Örnek giriş verisi (RGB formatında)
sample_input = torch.randn(1, 3, 512, 512).to(device)

# Modelin ilerletilmesi
model_output = model(sample_input)

print(model_output.shape)  # Model çıkış şeklini kontrol etmek için
#%%
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
#%%

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

smooth = 100

def dice_coef(y_true, y_pred):
    y_truef = torch.flatten(y_true)
    y_predf = torch.flatten(y_pred)
    intersection = torch.sum(y_truef * y_predf)
    return ((2 * intersection + smooth) / (torch.sum(y_truef) + torch.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    return -iou(y_true, y_pred)

def train(model, train_loader, test_loader, optimizer, num_epochs):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_dice = 0.0
        total_jaccard = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            threshold = 0.5
            binary_preds = (torch.sigmoid(output) > threshold).float()

            dice = dice_coef(binary_preds, labels)
            jaccard = iou(binary_preds, labels)

            total_dice += dice.item() * images.size(0)
            total_jaccard += jaccard.item() * images.size(0)

            pbar.set_postfix({'Loss': running_loss / (i + 1),
                              'Dice': total_dice / ((i + 1) * images.size(0)),
                              'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Dice Coef: {total_dice / len(train_loader.dataset)}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Jaccard Index: {total_jaccard / len(train_loader.dataset)}")

        # Her epoch sonunda test seti üzerinde değerlendirme yap
        test(model, test_loader, epoch, num_epochs)

    # Eğitim ve test sürecinin sonunda maskeleri görselleştir
    # visualize_best_worst_masks(train_loader, is_train=True)
    # visualize_best_worst_masks(test_loader, is_train=False)

def test(model, test_loader, epoch, num_epochs):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Örnek bir kayıp fonksiyonu, kullanımınıza göre değiştirebilirsiniz

    total_dice = 0.0
    total_jaccard = 0.0
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            threshold = 0.5
            binary_preds = (torch.sigmoid(output) > threshold).float()

            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

            dice = dice_coef(binary_preds, labels)
            jaccard = iou(binary_preds, labels)

            total_dice += dice.item() * images.size(0)
            total_jaccard += jaccard.item() * images.size(0)

            pbar.set_postfix({'Loss': total_loss / ((i + 1) * images.size(0)),
                              'Dice': total_dice / ((i + 1) * images.size(0)),
                              'Jaccard': total_jaccard / ((i + 1) * images.size(0))})

        print(f"Epoch [{epoch}/{num_epochs}] - Average Dice Coef: {total_dice / len(test_loader.dataset)}")
        print(f"Epoch [{epoch}/{num_epochs}] - Average Jaccard Index: {total_jaccard / len(test_loader.dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = num_epochs  # num_epochs değerini burada tanımladığınızı varsayalım
train(model, train_loader, test_loader, optimizer, num_epochs)

#%%

checkpoint_path = 'C:/Users/Froggremann/Desktop/data/model_foto.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    # Diğer gerektiğiniz bilgileri buraya ekleyebilirsiniz
}, checkpoint_path)

print(f'Model ağırlıkları kaydedildi: {checkpoint_path}')

#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Modeli oluşturun ve kaydedilmiş ağırlıkları yükleyin
in_channels = 3  # Giriş kanal sayısı
out_channels = 3  # Çıkış kanal sayısı, 3 sınıf için
model = UNet(in_channels, out_channels)  # Modelinizi tanımlayın (UNet, önceki model sınıfınız olsun)
checkpoint_path = 'C:/Users/Froggremann/Desktop/data/model_foto.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# İşaretleme yapılacak görüntüyü yükleyin
image_path = 'C:/Users/Froggremann/Desktop/data/0S0A6521_face1.jpg'
image = Image.open(image_path).convert('RGB')  # Girişi 3 kanal olarak açın
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Gerekirse boyutu değiştirin
    transforms.ToTensor(),
])
input_image = transform(image).unsqueeze(0)  # Batch boyutunu ekleyin

# Görüntüyü modelde değerlendirin
with torch.no_grad():
    output = model(input_image)

# İkili bir eşikleme uygulayarak maskeden bir tahmin elde edin
threshold = 0.5  # İkili eşik değeri
binary_prediction = (torch.sigmoid(output) > threshold).float()

# Giriş görüntüsünü RGB formatına çevirin
input_image_rgb = input_image.squeeze(0).permute(1, 2, 0).numpy()

# İkili tahmin maskesini boyutlarına uygun hale getirin
binary_prediction_rgb = binary_prediction.squeeze(dim=0).permute(1, 2, 0).numpy()

# Şeffaf kırmızı renkte bir maske oluşturun
overlay_mask = binary_prediction_rgb > 0.5  # Maske alanını belirleyin

# Maskeyi boyutlandırın ve RGB formatına çevirin
resized_mask = np.array(Image.fromarray((overlay_mask * 255).astype(np.uint8)).resize((512, 512))) > 0

# Maskeyi 3 boyutlu hale getirin
resized_mask_rgb = np.stack([resized_mask]*3, axis=-1)

# Maskeyi görüntünün üzerine ekleyin
overlay_image = input_image_rgb.copy()
# Maskeyi görüntünün üzerine ekleyin
mask_indices = np.argwhere(resized_mask_rgb[..., 0])  # True değerlerin indekslerini bulun
overlay_image[mask_indices[:, 0], mask_indices[:, 1], 0] = 1.0  # Kırmızı kanalı 0.0 yapın
overlay_image[mask_indices[:, 0], mask_indices[:, 1], 1] = 0.0  # Yeşil kanalı 1.0 yapın
overlay_image[mask_indices[:, 0], mask_indices[:, 1], 2] = 0.0  # Mavi kanalı 0.0 yapın

# Şeffaflığı artırmak için maske üzerine bir değer ekleyin
alpha = 0.1  # Şeffaflık değeri (0.0 ile 1.0 arasında)
rgba_overlay_image = np.concatenate([overlay_image, np.ones_like(overlay_image[..., :1]) * alpha], axis=-1)


# Sonuçları görselleştirin
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(input_image_rgb)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(binary_prediction_rgb, cmap='gray')
plt.title('Predicted Mask')

plt.subplot(1, 3, 3)
plt.imshow(overlay_image)
plt.title('Overlay (Original + Mask)')

plt.show()








