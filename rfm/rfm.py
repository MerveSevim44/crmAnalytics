###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)

df_ = pd.read_excel("C:/Users/merve/crmAnalytics/datasets/online_retail_II.xlsx",sheet_name="Year 2009-2010")
df = df_.copy()
print(df.head())
print(df.shape)
print(df.isnull().sum())



###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################
#eşsiz ürün sayısı
print(df["Description"].nunique())
print(df["Description"].value_counts().head())

print(df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity",ascending= False).head())
print(df["Invoice"].nunique())

df["Total_Price"] = df["Quantity"] * df["Price"]
print(df.head())
print(df.groupby("Invoice").agg({"Total_Price" : "sum"}).head())


###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################
print(df.shape)
print(df.isnull().sum())
df.dropna(inplace= True)
print(df.shape)

print(df.describe().T)
print(df[~df["Invoice"].str.contains("C",na = False)])
df = df[~df["Invoice"].str.contains("C",na = False)]

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency, Frequency, Monetary
print(df["InvoiceDate"].max())
today_date = dt.datetime(2010,12,11)
print(type(today_date))

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (today_date - x.max()).days,  # Recency hesaplama
    "Invoice": "nunique",  # Frequency hesaplama
    "Total_Price": "sum"  # Monetary hesaplama
})

print(rfm.head())

rfm.columns = ["Recency", "Frequency", "Monetary"]
print(rfm.head())

print(rfm.describe().T)

rfm = rfm[rfm["Monetary"] > 0]

print(rfm.shape)

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

rfm["recency_score"] = pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"],5,labels=[1,2,3,4,5])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method = "first"),5,labels=[1,2,3,4,5])

rfm["RFM_Score"] = (rfm["recency_score"].astype(str) + rfm["monetary_score"].astype(str) )

print(rfm.head())
print(rfm[rfm["RFM_Score"] == "55"])

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################


# RFM isimlendirmesi
# regex
seg_map = {
   r'[1-2][1-2]' : 'hibernating',
   r'[1-2][3-4]' : 'at_Risk',
   r'[1-2]5' : 'cant_loose',
   r'3[1-2]' : 'about_to_sleep',
   r'33' : 'need_attention',
   r'[3-4][4-5]' : 'loyal_costumers',
   r'41' : 'promising',
   r'51' : 'new_customers',
   r'[4-5][2-3]' : 'potential_loyalists',
   r'5[4-5]' : 'champions'

}

rfm["segment"] = rfm["RFM_Score"].replace(seg_map, regex = True)

print(rfm.head())

print(rfm[["segment","Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean","count"]))

print(rfm[rfm["segment"] == "need_attention"].head())
print(rfm[rfm["segment"] == "cant_loose"].index)


new_df = pd.DataFrame()
new_df["new_customer_Id"] = rfm[rfm["segment"] == "new_customers"].index

print(new_df["new_customer_Id"])
new_df["new_customer_Id"] = new_df["new_customer_Id"].astype(int)

print(new_df["new_customer_Id"])

new_df.to_csv("new_customer_Id.csv")


###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################

def create_rfm(dataframe,cvs = False):
    """
    Create RFM (Recency, Frequency, Monetary) analysis from transaction data.

    Parameters:
    dataframe: pandas DataFrame with columns ['Customer ID', 'Invoice', 'Quantity', 'Price', 'InvoiceDate']

    Returns:
    pandas DataFrame with RFM analysis results
    """
    # Create Total Price column
    dataframe["Total_Price"] = dataframe["Quantity"] * dataframe["Price"]

    # Data cleaning
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # Get analysis date (using the most recent date in the data plus 1 day)
    today_date = dataframe["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
    rfm = dataframe.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (today_date - x.max()).days,  # Recency
        "Invoice": "nunique",  # Frequency
        "Total_Price": "sum"  # Monetary
    })

    # Rename columns
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Filter out non-positive monetary values
    rfm = rfm[rfm["Monetary"] > 0]

    # Create RFM scores
    rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    # Combine R and F scores for segmentation
    rfm["RF_Score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # Define segment mapping
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_lose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    # Create segments
    rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

    # Calculate segment statistics
    segment_stats = rfm.groupby("segment").agg({
        "Recency": ["mean", "count"],
        "Frequency": "mean",
        "Monetary": "mean"
    })

    # Ensure Customer ID is integer type
    rfm.index = rfm.index.astype(int)

    if cvs:
        rfm.to_cvs("rfm_cvs")

    return rfm, segment_stats


df = df_.copy()

print(create_rfm(df))






