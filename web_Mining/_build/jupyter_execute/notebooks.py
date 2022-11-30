#!/usr/bin/env python
# coding: utf-8

# # **Data crawling**
# Data crawling adalah proses pengambilan data yang tersedia secara online untuk umum. Proses ini kemudian mengimpor informasi atau data yang telah ditemukan ke dalam file lokal di komputer Anda.

# ## Twint
# 
# Twint adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet.

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/web_mining/web_Mining')


# In[3]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# In[4]:


get_ipython().system('pip install nest-asyncio')


# In[5]:


get_ipython().system('pip install aiohttp==3.7.0')


# In[6]:


import nest_asyncio
nest_asyncio.apply() #digunakan sekali untuk mengaktifkan tindakan serentak dalam notebook jupyter.
import twint #untuk import twint
c = twint.Config()
c.Search = 'prabowo'
c.Pandas = True
c.Limit = 70
twint.run.Search(c)


# In[7]:


Tweets_dfs = twint.storage.panda.Tweets_df
Tweets_dfs["tweet"]


# In[8]:


Tweets_dfs["tweet"].to_csv("prabowo.csv")
from google.colab import files 
#files.download('prabowo.csv')


# # **Mengambil Data**
# 
# Proses ini digunakan untuk mengambil 100 data tweet yang telah disimpan dalam github dengan format .csv
# 
# 

# In[9]:


#install library pandas
get_ipython().system('pip install pandas')


# In[10]:


#install library numpy
get_ipython().system('pip install numpy')


# In[11]:


import pandas as pd 
import numpy as np

data_abstrak = pd.read_csv("https://raw.githubusercontent.com/SalmanAl27/webmining/master/prabowo.csv")

data_abstrak


# In[ ]:


#install library sastrawi
get_ipython().system('pip install sastrawi')


# In[ ]:


#install library swifter
get_ipython().system('pip install swifter')


# # **Case Folding**
# 
# Tahap untuk merubah teks yang memiliki huruf kapital menjadi huruf kecil

# In[ ]:


data_abstrak['abstrak'] = data_abstrak['abstrak'].str.lower()


data_abstrak['abstrak']


# In[ ]:


#install library nltk
get_ipython().system('pip install nltk')


# ## Menghapus Karakter Spesial

# In[ ]:


import string 
import re #regex library
# import word_tokenize & FreqDist from NLTK

from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# ------ Tokenizing ---------

def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_special)
data_abstrak['abstrak']


# ## Menghapus Angka

# In[ ]:



#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_number)
data_abstrak['abstrak']


# ## Menghapus Tanda Baca

# In[ ]:



#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_punctuation)
data_abstrak['abstrak']


# ## Menghapus Spasi
# 

# In[ ]:



#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_whitespace_multiple)
data_abstrak['abstrak']


# ## Menghapus huruf 

# In[ ]:



# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(remove_singl_char)
data_abstrak['abstrak']


# In[ ]:


import nltk
nltk.download('punkt')


# # **Tokenizing**
# 
# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token

# In[ ]:



# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(word_tokenize_wrapper)
data_abstrak['abstrak']


# # **Filtering(Stopwords Removal)**
# Proses untuk menghapus kata hubung atau kata yang tidak memiliki makna
# 

# In[ ]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[ ]:



list_stopwords = stopwords.words('indonesian')

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

data_abstrak['abstrak'] = data_abstrak['abstrak'].apply(stopwords_removal)

data_abstrak['abstrak']


# # **Stemming**
# Data training hasil dari filtering akan dilakukan pengecekan atau pencarian kata-kata yang sesuai dengan kamus umum. Apabila data training hasil filtering sesuai dengan kamus umum maka kata akan dikeluarkan sementara, karena sudah dianggap sebagai kata dasar. Apabila masih terdapat kata yang tidak termasuk dalam kata dasar maka tahap selanjutnya adalah menghapus inflection suffixes yang merupakan akhiran pertama. Kata yang memiliki akhiran partticles seperti “-pun”, “-kah”, “-tah”, “- lah” dan akhiran possessive pronoun seperti “-mu”, “-ku” dan “-nya” dihilangkan. Setelah dilakukan proses case folding, tokenezing, dan filtering, proses selanjutnya yaitu stemming. Stemming yang digunakan pada penelitian ini menggunakan algoritma Enhanced Confix Stipping Stemmer, terdiri dari beberapa langkah: Data training hasil dari filtering akan dilakukan pengecekan atau pencarian kata-kata yang sesuai dengan kamus umum. Apabila data training hasil filtering sesuai dengan kamus umum maka kata akan dikeluarkan sementara, karena sudah dianggap sebagai kata dasar. Apabila masih terdapat kata yang tidak termasuk dalam kata dasar maka tahap selanjutnya adalah menghapus inflection suffixes yang merupakan akhiran pertama. Kata yang memiliki akhiran partticles seperti “-pun”, “-kah”, “-tah”, “- lah” dan akhiran possessive pronoun seperti “-mu”, “-ku” dan “-nya” dihilangkan.

# In[ ]:


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in data_abstrak['abstrak']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")


# In[ ]:



for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

data_abstrak['abstrak'] = data_abstrak['abstrak'].swifter.apply(get_stemmed_term)
data_abstrak['abstrak']


# ##Menyimpan Hasil Tahap Preprocessing ke file .csv

# In[ ]:



#data_abstrak.to_csv('preprocessing.csv')


# # **TF**
# 
# TF(Term Frequency) : Istilah frekuensi kata dalam dokumen. Ada beberapa cara untuk menghitung frekuensi ini, dengan cara yang paling sederhana adalah dengan menghitung jumlah kata yang muncul dalam dokumen. Lalu, ada cara untuk menyesuaikan frekuensi, berdasarkan panjang dokumen, atau dengan frekuensi mentah kata yang paling sering muncul dalam dokumen.

# In[ ]:



from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('preprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['abstrak'])
dataTextPre


# In[ ]:


matrik_vsm=bag.toarray()
#print(matrik_vsm)
matrik_vsm.shape


# In[ ]:


matrik_vsm[0]


# In[ ]:


a=vectorizer.get_feature_names()


# In[ ]:


print(len(matrik_vsm[:,1]))
#dfb =pd.DataFrame(data=matrik_vsm,index=df,columns=[a])
dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
DataTFIDF = TfidfVectorizer()
TFIDF = DataTFIDF.fit_transform(dataTextPre['abstrak']).toarray()
TFIDF = pd.DataFrame(TFIDF)
TFIDF


# # KMeans(Clustering)
# 

# K-Means Clustering adalah suatu metode penganalisaan data atau metode Data Mining yang melakukan proses pemodelan unssupervised learning dan menggunakan metode yang mengelompokan data berbagai partisi.
# 
# K Means Clustering memiliki objective yaitu meminimalisasi object function yang telah di atur pada proses clasterisasi. Dengan cara minimalisasi variasi antar 1 cluster dengan maksimalisasi variasi dengan data di cluster lainnya.

# In[ ]:


from sklearn.cluster import KMeans

kmeans =KMeans(n_clusters=3)
kmeans=kmeans.fit(dataTF)
prediksi=kmeans.predict(dataTF)
centroids = kmeans.cluster_centers_

data=pd.DataFrame(prediksi,columns=["Cluster"])
data


# Menambah Label pada Hasil TF

# In[ ]:


datalabel = pd.read_csv('https://raw.githubusercontent.com/SalmanAl27/webmining/master/prabowo.csv')
dataJurnal = pd.concat([dataTF.reset_index(drop=True), datalabel["label"]], axis=1)
dataJurnal


# In[ ]:


dataJurnal['label'].unique()


# In[ ]:


dataJurnal.info()


# ### Split Data

# In[ ]:


### Train test split to avoid overfitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataJurnal.drop(labels=['label'], axis=1),
    dataJurnal['label'],
    test_size=0.3,
    random_state=0)


# In[ ]:


X_train


# # KNN (Analisis Sentimen)

# KNN adalah model sederhana untuk tugas-tugas regresi dan klasifikasi. Ketetanggan adalah representasi dari contoh pelatihan dalam ruang metrik. Ruang metrik adalah ruang fitur di mana jarak antara semua anggota set didefinisikan. Berkaitan dengan masalah pizza pada bab sebelumnya, contoh data latih diwakili dalam ruang metrik karena jarak antara semua diameter pizza ditentukan. Ketetanggan ini digunakan untuk memperkirakan nilai variabel respon untuk contoh uji. Hyperparameter k menentukan berapa banyak tetangga yang dapat digunakan dalam estimasi. Hyperparameter adalah parameter yang mengontrol bagaimana algoritma belajar; hiperparameter tidak diperkirakan dari data pelatihan dan kadang-kadang ditetapkan secara manual. Akhirnya, k tetangga yang dipilih adalah yang terdekat dengan instan uji, yang diukur dengan beberapa fungsi jarak.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
testing=[]
listnum=[]
for i in range(2,21):
  listnum.append(i)
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(X_train, y_train)
  Y_pred = neigh.predict(X_test) 
  testing.append(Y_pred)
testing


# In[ ]:


y_test


# ##hasil

# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score,precision_score
listtest=[]
listacc=[]
for i in range(len(testing)):
  accuracy_neigh=round(accuracy_score(y_test,testing[i])* 100, 2)
  acc_neigh = round(neigh.score(X_train, y_train) * 100, 2)
  listappend=listnum[i]
  appendlist=listappend,accuracy_neigh
  listtest.append(appendlist)
  listacc.append(accuracy_neigh)
listtest


# In[ ]:


from matplotlib import pyplot as plt
plt.bar(listnum, listacc)
plt.xticks(listnum)
plt.title('Nilai Akurasi Berdasarkan Input')
plt.ylabel('Persentase Akurasi')


# # Topic Modeling

# Tipic Modeling adalah cara mengelompokan data text berdasarkan suatu topik tertentu. Memiliki tujuan yang sama dengan klasifikasi tetapi menggunakan pendekatan berbeda topik modelling merupakan unsupervised learning alias tidak membutuhkan data berlabel. bisa dikatakan topic modelling bekerja seperti clustering dengan mengelompokan dokumen berdasarkan kemiripanya, tetapi topic modelling mempunyai tujuan yang lebih spesifik yaitu :
# 
# 1. Menemukan pola topik abstrak pada kumpulan dokumen
# 2. Memberikan anotasi dokumen berdasarkan topik tersebut
# 3. Menggunakan Anotasi dokumen untuk mengelompokan dokumen

# ## Information Gain

# Mutual Information atau Information Gain adalah salah satu metode dari seleksi fitur, dalam proses Information Gain fitur akan diranking, ranking fitur yang terbesar merupakan fitur yang paling relevan dan memiliki koneksi yang kuat dengan kumpulan data yang terkait 

# In[ ]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[ ]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[ ]:


#let's plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[ ]:


from sklearn.feature_selection import SelectKBest


# In[ ]:


#No we Will select the  top 5 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# Model LSA

# In[ ]:


from sklearn.decomposition import TruncatedSVD
#Membuat LSA Model
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)
#Ft Transform LSA Model dari TF-IDF
lsa_top=lsa_model.fit_transform(TFIDF)


# In[ ]:


#Menampilkan LSA top 10
print(lsa_top)
print(lsa_top.shape)


# Menampilkan Kekuatan Topik pada Setiap Dokumen

# In[ ]:


for Dokumen in range(TFIDF.shape[0]):
    print("\nDokumen : ", Dokumen)
    l=lsa_top[Dokumen]
    for i,topik in enumerate(l):
        print("Topik ",i," : ",topik*100)


# Mengidentifikasi Komponen Kata Setiap Topik

# In[ ]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# In[ ]:


# Kata penting dari setiap Topik
vocab = DataTFIDF.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    #Menampilkan 10 kata penting setiap topik
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topik "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# # Ensemble Learning
# 

# Metode ensemble atau metode ansamble adalah algoritma dalam pembelajaran mesin (machine learning) dimana algoritma ini sebagai pencarian solusi prediksi terbaik dibandingkan dengan algoritma yang lain karena metode ensemble ini menggunakan beberapa algoritma pembelajaran untuk pencapaian solusi prediksi yang lebih baik daripada algoritma yang bisa diperoleh dari salah satu pembelajaran algoritma kosituen saja. Tidak seperti ansamble statistika didalam mekanika statistika biasanya selalu tak terbatas. Ansemble Pembelajaran hanya terdiri dari seperangkat model alternatif yang bersifat terbatas, namun biasanya memungkinkan untuk menjadi lebih banyak lagi struktur fleksibel yang ada diantara alternatif model itu sendiri.
# 
# Evaluasi prediksi dari ensemble biasanya memerlukan banyak komputasi daripada evaluasi prediksi model tunggal (single model), jadi ensemble ini memungkinkan untuk mengimbangi poor learning algorithms oleh performasi lebih dari komputasi itu.

# In[ ]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

estimators = [
    ('rf1', RandomForestClassifier(n_estimators=10, random_state=40)),
    ('rf2', RandomForestClassifier(n_estimators=5, random_state=40))
    
    ]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
    )

clf.fit(X_train, y_train).score(X_test, y_test)


# In[ ]:


from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X.shape

