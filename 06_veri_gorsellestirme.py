import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Grafiklerin Estetik Ayarları ---
# Seaborn kütüphanesinin stilini ve renk paletini ayarlayarak daha güzel grafikler elde edelim.
sns.set_theme(style="whitegrid", palette="pastel")

# --- Veri Yükleme Fonksiyonları ---
def orjinal_veriyi_yukle():
    """Veritabanından ham ve orijinal veriyi çeker."""
    conn = sqlite3.connect('telekom.db')
    df = pd.read_sql_query("SELECT * FROM musteriler", conn)
    conn.close()
    return df

def islenmis_veriyi_yukle():
    """İşlenmiş özellikleri ve eğitilmiş modeli yükler."""
    X = pd.read_csv('data/islenmis_ozellikler.csv')
    model = joblib.load('en_iyi_musteri_kaybi_modeli.joblib')
    return X, model

# --- Görselleştirme Fonksiyonları ---
def gorsel_1_churn_dagilimi(df):
    """Müşteri kaybı oranını bir pasta grafiği ile gösterir."""
    plt.figure(figsize=(8, 8))
    df['Churn Label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                              labels=['Kalan Müşteri', 'Ayrılan Müşteri'],
                                              colors=['skyblue', 'salmon'])
    plt.title('Genel Müşteri Kaybı (Churn) Oranı', fontsize=16)
    plt.ylabel('') # y ekseni etiketini kaldır
    plt.savefig('images/01_churn_orani.png')
    print("Grafik 1: Churn oranı pasta grafiği 'images' klasörüne kaydedildi.")

def gorsel_2_sozlesmeye_gore_churn(df):
    """Sözleşme tipine göre müşteri kaybı dağılımını gösterir."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Contract', hue='Churn Label', data=df)
    plt.title('Sözleşme Tipine Göre Müşteri Kaybı Durumu', fontsize=16)
    plt.xlabel('Sözleşme Tipi', fontsize=12)
    plt.ylabel('Müşteri Sayısı', fontsize=12)
    plt.legend(title='Müşteri Durumu')
    plt.savefig('images/02_sozlesmeye_gore_churn.png')
    print("Grafik 2: Sözleşme tipine göre churn grafiği 'images' klasörüne kaydedildi.")

def gorsel_3_aylik_ucret_dagilimi(df):
    """Ayrılan ve kalan müşterilerin aylık ücret dağılımlarını karşılaştırır."""
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='Monthly Charges', hue='Churn Label', fill=True, common_norm=False)
    plt.title('Aylık Ücretlerin Müşteri Kaybına Etkisi', fontsize=16)
    plt.xlabel('Aylık Ücret (Monthly Charges)', fontsize=12)
    plt.ylabel('Yoğunluk', fontsize=12)
    plt.legend(title='Müşteri Durumu', labels=['Ayrılan Müşteri', 'Kalan Müşteri'])
    plt.savefig('images/03_aylik_ucret_dagilimi.png')
    print("Grafik 3: Aylık ücret dağılım grafiği 'images' klasörüne kaydedildi.")

def gorsel_4_ozellik_onemi(X, model):
    """Modelin en önemli bulduğu özellikleri bir çubuk grafiği ile gösterir."""
    # Modelin özellik önem skorlarını alalım
    onem_skorlari = pd.Series(model.feature_importances_, index=X.columns)
    
    plt.figure(figsize=(12, 8))
    onem_skorlari.nlargest(15).sort_values().plot(kind='barh', color='mediumseagreen')
    plt.title('Model İçin En Önemli 15 Özellik', fontsize=16)
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.ylabel('Özellikler', fontsize=12)
    plt.tight_layout() # Grafiğin kenarlara sığmasını sağlar
    plt.savefig('images/04_ozellik_onemi.png')
    print("Grafik 4: Özellik önemi grafiği 'images' klasörüne kaydedildi.")

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    print("Veri görselleştirme script'i başlıyor...")
    
    # Keşifsel analiz için orijinal veriyi yükle
    df_orjinal = orjinal_veriyi_yukle()
    
    # Model yorumlama için işlenmiş veriyi ve modeli yükle
    X_islenmis, best_model = islenmis_veriyi_yukle()
    
    # Fonksiyonları çağırarak grafikleri oluştur ve kaydet
    gorsel_1_churn_dagilimi(df_orjinal)
    gorsel_2_sozlesmeye_gore_churn(df_orjinal)
    gorsel_3_aylik_ucret_dagilimi(df_orjinal)
    gorsel_4_ozellik_onemi(X_islenmis, best_model)
    
    print("\nTüm grafikler başarıyla oluşturuldu ve 'images' klasörüne kaydedildi.")