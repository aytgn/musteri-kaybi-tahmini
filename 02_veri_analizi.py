import pandas as pd
import sqlite3

db_dosyasi = 'telekom.db'
tablo_adi = 'musteriler'

conn = sqlite3.connect(db_dosyasi)
print(f"'{db_dosyasi}' veritabanına başarıyla bağlanıldı.")


sorgu = f"SELECT * FROM {tablo_adi}"
print(f"'{sorgu}' sorgusu çalıştırılıyor...")
df = pd.read_sql_query(sorgu, conn)
print("Veri başarıyla DataFrame'e yüklendi.")

conn.close()
print(f"'{db_dosyasi}' veritabanı bağlantısı kapatıldı.")


print("\n--- VERİ SETİNE İLK BAKIŞ ---")
print("\n1. Veri Setinin İlk 5 Satırı (.head()):")
print(df.head())


print("\n2. Veri Setinin Yapısı (.info()):")
df.info()

print("\n3. Sayısal Verilerin Özeti (.describe()):")
print(df.describe())