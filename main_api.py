# ==============================================================================
# GEREKLİ KÜTÜPHANELERİ İÇERİ AKTARMA
# ==============================================================================
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, create_model
from typing import Optional, Dict, Any
import traceback # Hata detaylarını yakalamak için

# ==============================================================================
# UYGULAMA BAŞLANGIÇ AYARLARI (Sadece bir kere çalışır)
# ==============================================================================
print("API Başlatılıyor: Model ve ayarlar yükleniyor...")

try:
    # 1. Modeli yükle
    MODEL_DOSYASI = 'en_iyi_musteri_kaybi_modeli.joblib'
    model = joblib.load(MODEL_DOSYASI)

    # 2. Varsayılan değerleri ve veri tiplerini almak için eğitim verisini yükle
    X_train_data = pd.read_csv('data/islenmis_ozellikler.csv')
    default_values = X_train_data.median()
    
    # 3. Pydantic modelini (veri şablonunu) dinamik olarak oluştur
    fields_definition: Dict[str, Any] = {
        col: (Optional[float], None) for col in X_train_data.columns
    }
    DynamicCustomerFeatures = create_model('DynamicCustomerFeatures', **fields_definition)
    
    print("Model, varsayılan değerler ve veri şablonu başarıyla yüklendi.")

except FileNotFoundError as e:
    print(f"KRİTİK HATA: Gerekli bir dosya bulunamadı - {e}")
    print("Lütfen önceki script'leri çalıştırdığınızdan emin olun. API başlatılamıyor.")
    # Uygulamanın bu durumda başlamaması için çıkış yapabiliriz.
    exit()

# ==============================================================================
# FastAPI UYGULAMASI
# ==============================================================================
app = FastAPI(
    title="Dayanıklı Müşteri Kaybı Tahmin API'si",
    description="Eksik verileri akıllıca yöneten ve hataları yakalayan bir API.",
    version="1.2.0"
)

# ==============================================================================
# API UÇ NOKTALARI (ENDPOINTS)
# ==============================================================================

@app.get("/")
def read_root():
    """API'nin çalıştığını kontrol etmek için bir hoş geldin mesajı."""
    return {"status": "online", "message": "Müşteri Kaybı Tahmin API'si çalışıyor."}



@app.post("/predict")
def predict_churn(customer_data: DynamicCustomerFeatures):
    """
    Müşteri özelliklerini alarak churn (kayıp) tahmini yapar.
    """
    try:
        # 1. Gelen isteği DataFrame'e dönüştür ve eksik değerleri doldur.
        df = pd.DataFrame([customer_data.dict()])
        df.fillna(default_values, inplace=True)
        df = df[X_train_data.columns]
        df = df.astype(X_train_data.dtypes)

        # 2. Tahminleri yap.
        prediction_numpy = model.predict(df)[0]
        probability_numpy = model.predict_proba(df)[0][1]

        # 3. YENİ VE KRİTİK ADIM: Veri Tiplerini Standartlaştır
        # NumPy'dan gelen sonuçları standart Python tiplerine çeviriyoruz.
        # Bu, JSON'a sorunsuz bir şekilde dönüştürülmelerini garanti eder.
        prediction = str(prediction_numpy)
        churn_probability = float(probability_numpy)

        # 4. Sonucu döndür.
        return {
            "prediction": prediction,
            "churn_probability": churn_probability, # f-string formatlamasını kaldırdık, saf sayı daha iyidir
            "used_features": df.iloc[0].to_dict()
        }
        
    except Exception as e:
        print("HATA: Tahmin sırasında bir sorun oluştu.")
        print(f"Hata Detayı: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Tahmin yapılırken bir iç sunucu hatası oluştu. Hata: {e}"
        )

