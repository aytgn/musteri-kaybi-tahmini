import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# --- Adım 1: İşlenmiş Veriyi Yükleme ---
X = pd.read_csv('data/islenmis_ozellikler.csv')
y = pd.read_csv('data/hedef.csv', header=None).squeeze()
print("İşlenmiş veri dosyaları başarıyla yüklendi.")

# --- Adım 2: Veriyi Eğitim ve Test Setlerine Ayırma ---
# Stratify=y kullanarak sınıf dağılımını korumaya devam ediyoruz.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Veri, eğitim ve test setlerine ayrıldı.")

# --- Adım 3: Hiperparametre Izgarasını (Grid) Tanımlama ---
# Denemek istediğimiz hiperparametreleri ve alabilecekleri değerleri bir sözlük olarak tanımlıyoruz.
# GridSearchCV, buradaki tüm olası kombinasyonları (2*3*2*2 = 24 farklı model) deneyecektir.
# Not: Gerçek projelerde bu aralıklar daha geniş olabilir, ancak işlem süresi de artar.
param_grid = {
    'n_estimators': [150, 200],               # Ormandaki ağaç sayısı
    'max_depth': [10, 20, 30, None],              # Ağaçların maksimum derinliği (None = sınırsız)
    'min_samples_leaf': [1, 2, 4, 6],               # Bir yaprakta olması gereken minimum örnek sayısı
    'criterion': ['gini', 'entropy']          # Bölünme kalitesini ölçme kriteri
}

# --- Adım 4: GridSearchCV'yi Kurma ve Çalıştırma ---
# --- Fonksiyon/Sınıf Derinlemesine Bakış: GridSearchCV ---
# Kütüphane/Modül: sklearn.model_selection
# Amacı: Belirttiğimiz bir hiperparametre "ızgarası" (grid) içindeki tüm olası
# kombinasyonları, Çapraz Doğrulama (Cross-Validation) tekniği ile sistematik
# olarak deneyerek en iyi sonucu veren kombinasyonu bulur.
# Parametreleri:
#   - estimator: Ayarlarını optimize etmek istediğimiz model (boş bir RandomForestClassifier).
#   - param_grid: Denediğimiz hiperparametre sözlüğü.
#   - cv=5: Çapraz Doğrulama katman sayısı. Eğitim verisini 5'e böler, 4'ü ile eğitir, 1'i ile test eder
#     ve bunu 5 kez tekrarlar. Bu, sonucun daha güvenilir olmasını sağlar.
#   - scoring='recall_macro': GridSearchCV'ye "senin tek görevin, denediğin tüm
#     kombinasyonlar arasından Recall skorunu en yüksek yapanı bulmaktır" demektir.
#   - n_jobs=-1: Bilgisayarın tüm işlemci çekirdeklerini kullanarak aramayı hızlandırır.
#   - verbose=2: Arama sırasında ekrana detaylı bilgi yazdırarak süreci takip etmemizi sağlar.

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, 
                           scoring='recall_macro', n_jobs=-1, verbose=2)

print("\nGridSearchCV ile hiperparametre optimizasyonu başlıyor... (Bu işlem birkaç dakika sürebilir)")
# .fit() komutu, tanımlanan tüm kombinasyonları ve çapraz doğrulamayı çalıştırır.
grid_search.fit(X_train, y_train)

# --- Adım 5: En İyi Sonuçları ve Modeli Alma ---
print("\nOptimizasyon Tamamlandı!")
print("Bulunan En İyi Parametreler:", grid_search.best_params_)

# GridSearchCV, en iyi parametrelerle tüm eğitim verisi üzerinde zaten eğitilmiş olan modeli .best_estimator_ içinde tutar.
best_model = grid_search.best_estimator_

# --- Adım 6: En İyi Model ile Tahmin ve Değerlendirme ---
print("\n--- EN İYİ MODELİN DEĞERLENDİRME RAPORU ---")
y_pred = best_model.predict(X_test)

print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# --- Adım 7: En İyi Modeli Kaydetme ---
model_dosyasi = 'en_iyi_musteri_kaybi_modeli.joblib'
joblib.dump(best_model, model_dosyasi)

print(f"\nOptimize edilmiş en iyi model '{model_dosyasi}' olarak başarıyla kaydedildi.")