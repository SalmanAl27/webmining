#!/usr/bin/env python
# coding: utf-8

# # **UTS WEB MINING**

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


# %cd /content/drive/MyDrive/web_mining/


# # 1. Analisa Clustering dengan menggunakan K-Mean
# 
# 
# 

# # **Data crawling**
# Data crawling atau perayapan data adalah proses pengambilan data yang tersedia secara online untuk umum. Proses ini kemudian mengimpor informasi atau data yang telah ditemukan ke dalam file lokal di komputer Anda.

# ## Twint
# 
# Twint adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet.
# 
# Bukan hanya digunakan pada tweet, twint juga bisa kita gunakan untuk melakukan scrapping pada user, followers, retweet dan sebagainya. Twint memanfaatkan operator pencarian twitter untuk memungkinkan proses penghapusan tweet dari user tertentu, memilih dan memilah informasi-informasi yang sensitif, termasuk email dan nomor telepon di dalamnya.

# In[3]:


# !git clone --depth=1 https://github.com/twintproject/twint.git
# %cd twint
# !pip3 install -r requirements.txt


# In[4]:


# !pip install nest-asyncio


# In[5]:


# !pip install aiohttp==3.7.0


# In[6]:


import nest_asyncio
nest_asyncio.apply() #digunakan sekali untuk mengaktifkan tindakan serentak dalam notebook jupyter.
import twint #untuk import twint
c = twint.Config()
c.Search = 'tragedi Kanjuruhan'
c.Pandas = True
c.Limit = 70
twint.run.Search(c)


# In[111]:


Tweets_dfs = twint.storage.panda.Tweets_df
Tweets_dfs["tweet"]


# In[112]:


Tweets_dfs["tweet"].to_csv("Kanjuruhan.csv")


# # **Mengambil Data**
# 
# Proses ini digunakan untuk mengambil 100 data tweet yang telah disimpan dalam github dengan format .csv
# 
# 

# In[113]:


#install library pandas
get_ipython().system('pip install pandas')


# In[ ]:


#install library numpy
get_ipython().system('pip install numpy')


# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:


#install library sastrawi
get_ipython().system('pip install sastrawi')


# In[ ]:


#install library swifter
get_ipython().system('pip install swifter')


# # *Case Folding**
# 
# Tahap untuk merubah teks yang memiliki huruf kapital menjadi huruf kecil

# In[ ]:


Tweets_dfs["tweet"] = Tweets_dfs["tweet"].str.lower()


Tweets_dfs["tweet"]


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
                
Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_special)
Tweets_dfs["tweet"]


# ## Menghapus Angka

# In[ ]:



#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_number)
Tweets_dfs["tweet"]


# ## Menghapus Tanda Baca

# In[ ]:



#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_punctuation)
Tweets_dfs["tweet"]


# ## Menghapus Spasi
# 

# In[ ]:



#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_whitespace_multiple)
Tweets_dfs["tweet"]


# ## Menghapus huruf 

# In[ ]:



# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(remove_singl_char)
Tweets_dfs["tweet"]


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

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(word_tokenize_wrapper)
Tweets_dfs["tweet"]


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

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].apply(stopwords_removal)

Tweets_dfs["tweet"]


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

for document in Tweets_dfs["tweet"]:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

Tweets_dfs["tweet"] = Tweets_dfs["tweet"].swifter.apply(get_stemmed_term)
Tweets_dfs["tweet"]


# ##Menyimpan Hasil Tahap Preprocessing ke file .csv

# In[ ]:



Tweets_dfs.to_csv('preproKajuruhan.csv')


# # **TF**
# 
# TF(Term Frequency) : Istilah frekuensi kata dalam dokumen. Ada beberapa cara untuk menghitung frekuensi ini, dengan cara yang paling sederhana adalah dengan menghitung jumlah kata yang muncul dalam dokumen. Lalu, ada cara untuk menyesuaikan frekuensi, berdasarkan panjang dokumen, atau dengan frekuensi mentah kata yang paling sering muncul dalam dokumen.

# In[ ]:



from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('preproKajuruhan.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre["tweet"])


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


#  # 2. Peringkasan Berita 

# # KMeans

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


# # **MERINGKAS DOKUMEN**

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install scrapy\n!pip install pandas\n!pip install scikit-learn\n!pip install --user -U nltk\n\n!pip install scipy\n!pip install networkx')


# In[ ]:


get_ipython().system('rm -rf kompas_scrape.py hasil.json && sleep 1')


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a kompas_scrape.py', 'import string\nimport scrapy\nfrom scrapy import Request\n\n#@markdown ---\n#@markdown ### Masukkan url artikel kompas.com :\nartikel_url = "https://tekno.kompas.com/read/2022/10/10/14000087/kisah-aa-korban-pig-butchering-asal-indonesia-yang-rugi-rp-500-an-juta" #@param {type:"string"}\nartikel_url = artikel_url+"?page=all"\n#@markdown ---\n\nclass ptaUTM(scrapy.Spider):\n    name = "Kompas"\n    start_urls = [\n      artikel_url\n      ]\n\n\n    def parse(self, response):\n        a = ""\n        for paragraf in response.xpath(\'//div[contains(@class,"clearfix")]/p/text()\').getall():\n          a = a+" "+str(paragraf)\n        item = {\n            \'judul\' : response.xpath(\'//h1[contains(@class,"read__title")]/text()\').get(),\n            \'konten\' : a\n        }\n\n        yield item')


# ## **Menjalankan Script Scrapy**
# Setelah sebelumnya kita menulis script konfigurasi scrapy, selanjutnya kita akan menjalankan script tersebut dengan perintah "scrapy runspider" yang diikuti dengan nama berkas script dan dan nama berkas output untuk menyimpan hasilnya.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!scrapy runspider kompas_scrape.py -O hasil.json && sleep 2')


# ## **Import Hasil**
# Setelah perintah sebelumnya berhasil dijalankan, selanjutnya kita akan melakukan import isi berkas hasil scrape sebelumnya kedalam Pandas DataFrame.

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df_scrape = pd.read_json('hasil.json')
df_scrape


# In[ ]:


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

df_pisahkalimat = df_scrape.copy()
df_pisahkalimat ["konten"][0] = sent_tokenize(df_pisahkalimat["konten"][0])

df_pisahkalimat = pd.DataFrame(df_pisahkalimat["konten"][0], columns=["kalimat"])
df_pisahkalimat


# In[ ]:


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

df_pisahkalimat = df_scrape.copy()
df_pisahkalimat ["konten"][0] = sent_tokenize(df_pisahkalimat["konten"][0])

df_pisahkalimat = pd.DataFrame(df_pisahkalimat["konten"][0], columns=["kalimat"])
df_pisahkalimat


# ### Case Folding
# Case folding merupakan tahap text preprocessing yang berguna untuk :
# 
# 
# *   Mengubah huruf kapital menjadi huruf kecil
# *   Mengapus tanda baca
# *   Menghapus angka
# *   Menghapus karakter kosong
# 
# 
# 

# In[ ]:


import string
df_casefolding = df_pisahkalimat.copy()
#mengubah menjadi huruf kecil
df_casefolding['kalimat'] = df_casefolding['kalimat'].str.lower()

#menghapus tanda baca
tanda_baca = string.punctuation
tanda_baca = tanda_baca+"–"
for char in tanda_baca:
    df_casefolding['kalimat'] = df_casefolding['kalimat'].replace(r'[\%s]'%char," ", regex=True)

#menghapus angka
df_casefolding['kalimat'] = df_casefolding['kalimat'].replace(r'\d+',' ', regex=True)

#menghapus karakter kosong
df_casefolding['kalimat'] = df_casefolding['kalimat'].replace(r'\s+',' ', regex=True)
df_casefolding.head(2)


# ### Vectorization
# 
# Dalam Machine Learning, vektorisasi adalah langkah dalam ekstraksi fitur. Idenya adalah untuk mendapatkan beberapa fitur berbeda dari teks untuk model untuk dilatih, dengan mengubah teks menjadi vektor numerik.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(df_casefolding['kalimat'])


# In[ ]:


print ("Banyaknya kosa kata = ", len((cv.get_feature_names_out())))


# In[ ]:


print ("kosa kata = ", (cv.get_feature_names_out()))


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())
print(normal_matrix.T.toarray)


# ### Cosine Similarity
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np
vektorkalimat=normal_matrix.toarray()
A=vektorkalimat[0]
B=vektorkalimat[1]
dot = np.dot(A, B)
norma = np.linalg.norm(A)
normb = np.linalg.norm(A)
cos = dot / (norma * normb)
cos


# ## Membangun Graph
# 
# Graph merupakan sekumpulan objek terstruktur di mana beberapa pasangan objek mempunyai hubungan ataupun keterkaitan tertentu. Graph disini digunakan untuk melihat seberapa banyak hubungan antara kalimat satu dengan lainnya.

# In[ ]:


res_graph = normal_matrix * normal_matrix.T
print(res_graph)


# In[ ]:


import networkx as nx
nx_graph = nx.from_scipy_sparse_matrix(res_graph)


# In[ ]:


nx.draw_circular(nx_graph)


# ## Pagerank
# 
# PageRank adalah algoritma analisis tautan dan memberikan bobot numerik untuk setiap elemen dari kumpulan dokumen yang ditautkan. 
# 
# Perhitungan PageRank memerlukan beberapa lintasan, yang disebut "iterasi", melalui koleksi untuk menyesuaikan perkiraan nilai PageRank agar lebih mencerminkan nilai teoretis yang sebenarnya.

# In[ ]:


# scores = nx.pagerank(nx_graph, alpha=0.88, max_iter=1000, )
# scores


# In[ ]:


#@title Persentase Ringkasan
#@markdown ---
#@markdown ### Tentukan persentase ringkasan : total kalimat:
persentase_ringkasan = 60 #@param {type:"slider", min:50, max:90, step:10}
#@markdown ---


# Mencocokkan indeks kalimat asli dan kalimat hasil preprocessing

# In[ ]:


# top_sentence={sentence:scores[index] for index,sentence in enumerate(df_pisahkalimat["kalimat"].values)}

# persentase = len(df_pisahkalimat["kalimat"])*persentase_ringkasan//100
# print("Jumlah Kalimat pada Ringkasan : ",persentase)
# top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:persentase])


# ### Hasil dan perbandingan

# In[ ]:


# ringkasan = ""
# for sent in df_pisahkalimat["kalimat"].values:
#     if sent in top.keys():
#         ringkasan = ringkasan+" "+sent

# #ringkasan
# perbandingan = {'Teks asli' : df_scrape["konten"], 'Hasil Ringkasan' : [ringkasan]}
# df_perbandingan = pd.DataFrame(perbandingan)
# df_perbandingan

