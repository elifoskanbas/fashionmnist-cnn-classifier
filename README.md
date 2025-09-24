# FashionMNIST CNN Eğitimi ve Değerlendirme

Bu proje, **FashionMNIST** veri seti üzerinde bir **Convolutional Neural Network (CNN)** modeli eğitmek, doğrulamak ve test etmek için hazırlanmıştır. Kod, erken durdurma (Early Stopping) ve en iyi modelin kaydedilmesini (Checkpoint) destekler.

## Kaggle Notebook Link:
-https://www.kaggle.com/code/elifoskanba/fashion-mnist-cnn
---

## Özellikler
- **Veri Seti:** Fashion-MNIST
- **Kategoriler:**
  - T-shirt/top
  - Trouser (Pantolon)
  - Pullover (Kazak)
  - Dress (Elbise)
  - Coat (Ceket)
  - Sandal
  - Shirt (Gömlek)
  - Sneaker (Spor Ayakkabı)
  - Bag (Çanta)
  - Ankle boot (Bilek Botu)

- **Veri Yükleme:** CSV formatındaki FashionMNIST veri seti kullanılır. `label` sütunu sınıf etiketlerini içerir.

- **Veri Dönüşümleri (Transforms):**
  - Görüntüler 32x32 boyutuna yeniden boyutlandırılır.
  - Tensör formatına dönüştürülür ve normalize edilir.
  - Eğitim setine ek veri artırma uygulanır (dönme, yatay çevirme).

- **Model:** 2 convolution + pooling katmanı, 1 dense katman, dropout, 10 sınıf çıkışı.

- **Eğitim:**
  - Loss: `CrossEntropyLoss`
  - Optimizer: `Adam` (lr=0.001)
  - Epoch: 20 (erken durdurma ile)
  - Early stopping: 3 epoch boyunca iyileşme yoksa durdurma
- **Değerlendirme:**
  - Test doğruluğu (Accuracy)
  - Confusion matrix
  - Classification report

---

## Kullanım

### 1. Veri Yükleme
```python
train_csv = "/kaggle/input/fashionmnist/fashion-mnist_train.csv"
test_csv  = "/kaggle/input/fashionmnist/fashion-mnist_test.csv"

train_data = FashionMNISTDataset(train_csv, transform=transform)
test_data  = FashionMNISTDataset(test_csv, transform=transform)
```

### 2. DataLoader
```python
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)
```

### 3. Model Eğitimi
```python
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### 4. Eğitim ve Doğrulama Döngüsü
-Model her epoch sonunda doğrulama kaybına göre kontrol edilir.
-En iyi model best_model.pth olarak kaydedilir.
-Early stopping ile gereksiz eğitim önlenir.

### 5. Performans Görselleştirme
```python
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss")
plt.show()
```

### 6. Test Değerlendirmesi
```python
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {test_acc:.4f}")
```

### 7. Confusion Matrix ve Rapor
```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
print(classification_report(all_labels, all_preds))
```

### Gereksinimler
-Python 3.x
-pandas
-numpy
-torch (PyTorch)
-torchvision
-matplotlib
-seaborn
-scikit-learn
-Pillow (PIL)

### Notlar
-Kod GPU kullanabiliyorsa otomatik olarak GPU kullanır.
-Data augmentation yalnızca eğitim veri setine uygulanır.
-Early stopping en iyi doğrulama kaybı noktasındaki ağırlıkları kaydeder.