# CRM Analytics Projects

Bu repo, müşteri analitiği (Customer Analytics) alanında kullanılan temel veri bilimi yöntemlerini uygulamalı olarak göstermektedir.

Projede aşağıdaki analizler yer almaktadır:

* RFM analizi
* CLTV hesaplama
* CLTV prediction (BG/NBD & Gamma-Gamma)
* Association Rule Learning
* Müşteri segmentasyonu

Amaç, e-ticaret veri setleri üzerinde müşteri davranışlarını analiz etmek ve pazarlama stratejilerine veri odaklı karar desteği sağlamaktır.

---

## Kullanılan Teknolojiler

Projede kullanılan başlıca kütüphaneler:

* pandas
* numpy
* lifetimes
* scikit-learn
* mlxtend
* matplotlib

Kurulum:

```bash
pip install pandas numpy lifetimes scikit-learn mlxtend matplotlib openpyxl
```

---

## Proje Yapısı

```
crmAnalytics/
│
├── rfm/
│   ├── rfm.py
│   └── new_customer_Id.csv
│
├── cltv/
│   └── cltv.py
│
├── cltv_prediction/
│   ├── cltv.py
│   └── cltv_prediction.py
│
├── bonus_arl.py
└── README.md
```

---

## 1. RFM Analizi

RFM analizi müşterileri şu metriklere göre segmentlere ayırır:

* Recency → Son alışveriş zamanı
* Frequency → Alışveriş sayısı
* Monetary → Harcama miktarı

Segmentler:

* Champions
* Loyal Customers
* Potential Loyalists
* At Risk
* Hibernating

Bu analiz sayesinde müşterilere özel kampanyalar planlanabilir.

---

## 2. CLTV Hesaplama

CLTV (Customer Lifetime Value), müşterinin şirket için toplam değerini tahmin etmek için kullanılır.

Hesaplanan metrikler:

* Average Order Value
* Purchase Frequency
* Profit Margin
* Customer Value
* CLTV

Müşteriler CLTV değerlerine göre segmentlere ayrılmıştır.

---

## 3. CLTV Prediction

Bu bölümde ileri seviye müşteri değeri tahmini yapılmıştır.

Kullanılan modeller:

* BG/NBD → satın alma sayısı tahmini
* Gamma-Gamma → ortalama kazanç tahmini

Sonuç:

* 3 ve 6 aylık satış tahmini
* Beklenen müşteri değeri
* CLTV segmentasyonu

---

## 4. Association Rule Learning

Market Basket Analysis uygulanmıştır.

Adımlar:

* Veri ön işleme
* Sepet matrisi oluşturma
* Apriori algoritması
* Association Rules
* Ürün öneri sistemi

Örnek kullanım:

```python
arl_recommender(rules, 22492, 3)
```

---

## Veri Setleri

Projelerde kullanılan veri setleri:

* Online Retail II Dataset
* FLO OmniChannel Dataset

Bu veri setleri e-ticaret işlemlerini ve müşteri davranışlarını içermektedir.

---

## Projeyi Çalıştırma

Örnek:

```bash
python rfm/rfm.py
```

```bash
python cltv_prediction/cltv_prediction.py
```

```bash
python bonus_arl.py
```

---

## Amaç

Bu repo aşağıdaki konuları uygulamalı olarak öğrenmek amacıyla hazırlanmıştır:

* Customer Analytics
* RFM Analysis
* CLTV Prediction
* Recommender Systems
* Segmentation

---

## Gelecek Çalışmalar

* Dashboard geliştirme
* Web tabanlı müşteri analitiği paneli
* Model sonuçlarının görselleştirilmesi

---

## Lisans

Eğitim amaçlı kullanım.
