import pandas as pd
import sqlite3

# --- Adım 1: Veriyi Veritabanından Yükleme ---
# Bu adımı her scriptin başında tekrarlayarak scriptleri bağımsız hale getiriyoruz.
db_dosyasi = 'telekom.db'
tablo_adi = 'musteriler'
conn = sqlite3.connect(db_dosyasi)
df = pd.read_sql_query(f"SELECT * FROM {tablo_adi}", conn)
conn.close()

print("Veri veritabanından yüklendi. Orijinal boyut:", df.shape)

# --- Adım 2: Gereksiz ve Sızıntı Yaratan Sütunları Atma ---
atılacak_sutunlar = [
    'CustomerID', 'Count', 'Country', 'State', 'Lat Long',
    'Churn Value', 'Churn Score', 'Churn Reason'
]
df_temiz = df.drop(columns=atılacak_sutunlar)
print("Gereksiz sütunlar atıldı. Yeni boyut:", df_temiz.shape)

# --- Adım 3: 'TotalCharges' Sütununu Sayısala Çevirme ---
# .to_numeric fonksiyonu metni sayıya çevirmeye çalışır.
# errors='coerce' parametresi: Eğer bir değeri çeviremezse (örneğin boşluksa),
# onu NaN (Not a Number - Boş Değer) olarak işaretler.
df_temiz['Total Charges'] = pd.to_numeric(df_temiz['Total Charges'], errors='coerce')

# Oluşan boş değerleri (NaN), sütunun medyanı (ortanca değeri) ile dolduruyoruz.
# inplace=True, değişikliği doğrudan DataFrame üzerinde yapar.
df_temiz['Total Charges'].fillna(df_temiz['Total Charges'].median(), inplace=True)
print("'Total Charges' sütunu sayısala çevrildi ve boş değerler dolduruldu.")


# --- Adım 4: Hedef Değişkeni (y) ve Özellikleri (X) Ayırma ---
hedef_sutun = 'Churn Label'
y = df_temiz[hedef_sutun]
X = df_temiz.drop(columns=[hedef_sutun])


# --- Adım 5: Kategorik Özellikleri Sayısala Çevirme (One-Hot Encoding) ---
# --- Fonksiyon Derinlemesine Bakış: pd.get_dummies() ---
# Kütüphane/Modül: pandas
# Amacı: Metin içeren kategorik sütunları, her bir kategori için 0 ve 1'lerden
# oluşan yeni "sanal" (dummy) sütunlara dönüştürür.
# Örneğin, 'Gender' sütununu 'Gender_Male' ve 'Gender_Female' gibi iki yeni
# sütuna ayırır. Bir müşteri erkekse 'Gender_Male' 1, diğeri 0 olur.
# drop_first=True: Oluşturulan sanal sütunlardan ilkini atar. Bu, gereksiz
# tekrarı önler (bir müşteri 'Male' değilse, zaten 'Female' olduğu bellidir).
print("Kategorik veriler sayısala çevriliyor...")
X_encoded = pd.get_dummies(X, drop_first=True)
print("Çevrim tamamlandı. Yeni özellik matrisinin boyutu:", X_encoded.shape)

# Sonucu kontrol etmek için ilk birkaç satırı görelim
print("\nİşlenmiş ve Genişletilmiş Veri Setinin İlk 5 Satırı:")
print(X_encoded.head())

# --- Adım 6: İşlenmiş Veriyi Kaydetme ---
# Bir sonraki adımda kullanmak üzere temiz verilerimizi dosyalara kaydediyoruz.
# index=False, pandas index'inin dosyaya yazılmasını engeller.
X_encoded.to_csv('data/islenmis_ozellikler.csv', index=False)
y.to_csv('data/hedef.csv', index=False, header=False)

print("\nİşlenmiş özellikler ve hedef verisi 'data' klasörüne kaydedildi.")