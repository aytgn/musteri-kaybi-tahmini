import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Adım 1: İşlenmiş Veriyi Yükleme ---
X = pd.read_csv('data/islenmis_ozellikler.csv')
# hedef.csv tek sütunlu ve başlıksız olduğu için özel parametrelerle okuyoruz.
# .squeeze(), tek sütunlu DataFrame'i bir Series'e dönüştürür.
y = pd.read_csv('data/hedef.csv', header=None).squeeze()

print("İşlenmiş veri dosyaları başarıyla yüklendi.")
print("Özellik matrisi boyutu:", X.shape)
print("Hedef vektörü boyutu:", y.shape)


# --- Adım 2: Veriyi Eğitim ve Test Setlerine Ayırma ---
# --- Yeni Parametre Derinlemesine Bakış: stratify=y ---
# Amacı: Veriyi bölerken, eğitim ve test setlerindeki hedef (y) dağılımının
# (örn: 'Yes'/'No' oranının) orijinal veri setiyle aynı olmasını sağlar.
# Bu, özellikle dengesiz veri setlerinde modelin daha doğru değerlendirilmesi için kritiktir.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nVeri, eğitim ve test setlerine ayrıldı.")


# --- Adım 3: Modeli Oluşturma ve Eğitme ---
# --- Sınıf Derinlemesine Bakış: RandomForestClassifier ---
# Kütüphane/Modül: sklearn.ensemble
# Amacı: Güçlü bir sınıflandırma modeli oluşturur. Bunu, eğitim sırasında yüzlerce
# basit "Karar Ağacı" oluşturup, bir tahmin yaparken bu ağaçların çoğunluğunun
# "oyunu" alarak yapar. Bu "kolektif akıl" yöntemi, tek bir ağacın yapabileceği
# hataları azaltır ve genellikle çok daha yüksek başarı sağlar.
# n_estimators=100: Ormanda oluşturulacak ağaç sayısı.
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Random Forest modeli eğitiliyor...")
model.fit(X_train, y_train)
print("Model başarıyla eğitildi.")


# --- Adım 4: Test Seti Üzerinde Tahmin Yapma ---
y_pred = model.predict(X_test)
print("\nTest seti üzerinde tahminler yapıldı.")


# --- Adım 5: Model Performansını Değerlendirme ---
print("\n--- MODEL DEĞERLENDİRME RAPORU ---")

# a) Accuracy (Doğruluk)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin Doğruluk Oranı (Accuracy): {accuracy:.4f}")

# b) Confusion Matrix (Karmaşıklık Matrisi)
#       Tahmin: No | Tahmin: Yes
# Gerçek: No  [ TN ]     [ FP ]
# Gerçek: Yes [ FN ]     [ TP ]
print("\nKarmaşıklık Matrisi (Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("(TN: Gerçekten 'No' olan ve 'No' tahmin edilenler)")
print("(FP: Gerçekten 'No' olan ama 'Yes' tahmin edilenler - Hatalı Alarm)")
print("(FN: Gerçekten 'Yes' olan ama 'No' tahmin edilenler - Gözden Kaçanlar)")
print("(TP: Gerçekten 'Yes' olan ve 'Yes' tahmin edilenler)")

# c) Classification Report (Detaylı Sınıflandırma Raporu)
# Precision, Recall, F1-score gibi daha detaylı metrikler sunar.
print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


# --- Adım 6: Eğitilmiş Modeli Kaydetme ---
model_dosyasi = 'musteri_kaybi_modeli.joblib'
joblib.dump(model, model_dosyasi)

print(f"\nEğitilmiş model '{model_dosyasi}' olarak başarıyla kaydedildi.")