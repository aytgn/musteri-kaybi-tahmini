import pandas as pd
import sqlite3

excel_dosyasi = 'data/churn_data.xlsx' 
db_dosyasi = 'telekom.db'
tablo_adi = 'musteriler'

print(f"'{excel_dosyasi}' okunuyor...")
df = pd.read_excel(excel_dosyasi)
print("Excel dosyası başarıyla DataFrame'e yüklendi.")

conn = sqlite3.connect(db_dosyasi)
print(f"'{db_dosyasi}' veritabanı bağlantısı açıldı.")

print(f"DataFrame, '{tablo_adi}' tablosuna yazılıyor...")
df.to_sql(tablo_adi, conn, if_exists='replace', index=False)
print("Veri, veritabanına başarıyla yazıldı.")

conn.close()
print(f"'{db_dosyasi}' veritabanı bağlantısı kapatıldı.")
print("\nİşlem Tamamlandı!")