##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns",None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv("C:/Users/merve/FLOCLTVPrediction/flo_data_20k.csv")

df = df_.copy()

print(df.isnull().sum())

print(df.describe().T)


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable, low_ratio=0.01, up_ratio=0.99):
   quartile1 = dataframe[variable].quantile(low_ratio)
   quartile3 = dataframe[variable].quantile(up_ratio)
   interquantile_range = quartile3 - quartile1
   up_limit = quartile3 + interquantile_range * 1.5
   low_limit = quartile1 - interquantile_range * 1.5

   if pd.notna(up_limit) and pd.notna(low_limit):
      return round(low_limit), round(up_limit)
   else:
      return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
   low_limit, up_limit = outlier_thresholds(dataframe, variable)

   dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
   dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit




# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

# Optional: Print out the thresholds before replacement
for var in ["order_num_total_ever_online",
            "order_num_total_ever_offline",
            "customer_value_total_ever_offline",
            "customer_value_total_ever_online"]:
    low, high = outlier_thresholds(df, var)
    print(f"{var}: Low Limit = {low}, High Limit = {high}")

# After replacement, compare descriptive statistics
print("Before replacement:")
print(df[["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]].describe().T)

# Perform replacements
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

print("\nAfter replacement:")
print(df[["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]].describe().T)


print(df.describe().T)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_expenditure"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["total_price"] = df["order_num_total_ever_offline"]*df["customer_value_total_ever_offline"] + df["order_num_total_ever_online"] * df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df["first_order_date"] = pd.to_datetime(df["first_order_date"], errors='coerce')
df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors='coerce')
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"], errors='coerce')
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"], errors='coerce')

# Check for any NaT values
print(df[["first_order_date", "last_order_date",
          "last_order_date_online", "last_order_date_offline"]].isna().sum())

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

# En son alışveriş tarihini bul ve 2 gün ekle
today_date = df["last_order_date"].max() + pd.Timedelta(days=2)

# CLTV için gerekli verileri hesapla

cltv_c = df.groupby("master_id").agg({
    "last_order_date": [
        lambda x: (today_date - x.max()).days,  # Recency
        lambda x: (today_date - x.min()).days   # T
    ],
    "total_order": "sum",
    "total_price": "sum"
})


cltv_c.columns = ["recency_cltv_c_weekly", "T_weekly", "frequency", "monetary_cltv_c_avg"]
cltv_c = cltv_c.reset_index()


cltv_c = cltv_c.astype({
    "frequency": float,
    "recency_cltv_c_weekly": float,
    "T_weekly": float,
    "monetary_cltv_c_avg": float
})

# Handle outliers
for col in ["frequency", "recency_cltv_c_weekly", "T_weekly", "monetary_cltv_c_avg"]:
    replace_with_thresholds(cltv_c, col)

# Filter out zero values
cltv_c = cltv_c[
    (cltv_c["frequency"] > 0) &
    (cltv_c["recency_cltv_c_weekly"] > 0) &
    (cltv_c["T_weekly"] > 0)
]


cltv_c["monetary_cltv_c_avg"] = cltv_c["monetary_cltv_c_avg"] / cltv_c["frequency"]


cltv_c["recency_cltv_c_weekly"] /= 7
cltv_c["T_weekly"] /= 7

# Verification checks
print("Column Types:")
print(cltv_c.dtypes)

print("\nDescriptive Statistics:")
print(cltv_c.describe())

print("\nNull Values:")
print(cltv_c[["frequency", "recency_cltv_c_weekly", "T_weekly"]].isnull().sum())

print("\nFirst few rows:")
print(cltv_c.head())

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık cltv_c'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

bgf = BetaGeoFitter(penalizer_coef= 0.01)

bgf.fit(cltv_c["frequency"],
        cltv_c["recency_cltv_c_weekly"],
        cltv_c["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv_c dataframe'ine ekleyiniz.

cltv_c["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                    cltv_c["frequency"],
                    cltv_c["recency_cltv_c_weekly"],
                    cltv_c["T_weekly"])


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv_c dataframe'ine ekleyiniz.

cltv_c["exp_sales_6_month"]= bgf.conditional_expected_number_of_purchases_up_to_time(24,
                    cltv_c["frequency"],
                    cltv_c["recency_cltv_c_weekly"],
                    cltv_c["T_weekly"])



# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
print("\nSummary Statistics:")
print(f"Total expected purchases in 3 month: {cltv_c["exp_sales_3_month"].sort_values(ascending=False).head(10)}")
print(f"Total expected purchases in 6 months: {cltv_c["exp_sales_6_month"].sort_values(ascending=False).head(10)}")

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv_c dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef= 0.01)
ggf.fit(cltv_c["frequency"],cltv_c["monetary_cltv_c_avg"])
cltv_c["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_c["frequency"],
                                                                             cltv_c["monetary_cltv_c_avg"]).sort_values(ascending=False)

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_c["frequency"],
                                   cltv_c["recency_cltv_c_weekly"],
                                   cltv_c["T_weekly"],
                                   cltv_c["monetary_cltv_c_avg"],
                                   time = 6,# 3 aylık
                                   freq = "W", #T nin frekans bilgisi
                                   discount_rate=0.01)
# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv = cltv.reset_index()
print(cltv.info)

cltv_final = pd.merge(cltv_c,cltv, on = "master_id",how = "left")

print(cltv_final.sort_values(by="clv", ascending=False).head(20))

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_final["cltv_segment"] = pd.qcut(cltv["clv"],4,["D","C","B","A"])

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

print(cltv_final.columns)

print(cltv_final.groupby("cltv_segment")[["frequency", "monetary_cltv_c_avg"]].agg("mean"))






