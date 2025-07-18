# Müşteri Kaybı Tahmini (Customer Churn Prediction) Projesi

Bu proje, bir telekomünikasyon şirketinin müşteri verilerini kullanarak, bir müşterinin şirketten ayrılıp ayrılmayacağını (`Churn`) tahmin eden bir makine öğrenmesi modelini içerir. Proje, veri ön işleme, model eğitimi ve değerlendirme adımlarını kapsamaktadır.

## Proje Akışı

Proje, modüler bir yaklaşımla birkaç script'e bölünmüştür:

1.  **`01_veritabani_olustur.py`**: Ham `.xlsx` veri setini okur ve verileri kalıcı bir `telekom.db` SQLite veritabanına yazar.
2.  **`02_veri_analizi.py`**: Veritabanından veriyi çeker ve temel keşifsel veri analizi (EDA) yapar.
3.  **`03_veri_onisleme.py`**: Veriyi temizler, gereksiz sütunları atar ve kategorik verileri modelin anlayabileceği sayısal formata dönüştürür.
4.  **`04_model_egitimi.py`**: İşlenmiş veriyi kullanarak bir `RandomForestClassifier` modeli eğitir, performansını değerlendirir ve eğitilmiş modeli kaydeder.

## Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın:** `git clone <repo_url>`
2.  **Proje Dizinine Gidin:** `cd musteri-kaybi-tahmini`
3.  **Sanal Ortamı Kurun ve Aktifleştirin:** `python -m venv venv` ve `venv\Scripts\activate`
4.  **Bağımlılıkları Yükleyin:** `pip install -r requirements.txt`
5.  **Script'leri Sırasıyla Çalıştırın:**
    ```bash
    python 01_veritabani_olustur.py
    python 02_veri_analizi.py
    python 03_veri_onisleme.py
    python 04_model_egitimi.py
    ```

## Model Performansı

Model, test seti üzerinde aşağıdaki sonuçları elde etmiştir:

* **Doğruluk (Accuracy):** 0.7963
* **Temel Bulgular:** Model, genel olarak başarılı olsa da, asıl hedefimiz olan ayrılacak müşterileri ("Yes" sınıfı) tespit etmede **Recall (Yakalama Oranı)** metriği *[Buraya Yes sınıfının Recall değerini yazın, örn: 0.53]* olarak ölçülmüştür. Bu, ayrılacak her 100 müşteriden 53'ünü doğru tespit edebildiğimiz anlamına gelir. Modelin en büyük zayıflığı, riskli müşterileri gözden kaçırmasıdır (False Negatives).

## Kullanılan Teknolojiler
- Python 3
- Pandas
- Scikit-learn
- SQLite
- Git & GitHub