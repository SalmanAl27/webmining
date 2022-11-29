#!/usr/bin/env python
# coding: utf-8

# # **Data crawling**
# Data crawling adalah proses pengambilan data yang tersedia secara online untuk umum. Proses ini kemudian mengimpor informasi atau data yang telah ditemukan ke dalam file lokal di komputer Anda.

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/web_mining/web_Mining')


# ## Twint
# 
# Twint adalah sebuah tools yang digunakan untuk melakukan scrapping dari aplikasi twitter yang disetting secara khusus menggunakan bahasa pemrograman Python. Twint dapat kita gunakan dan jalankan tanpa harus menggunakan API dari Twitter itu sendiri, dengan kapasitas scrapping data maksimalnya adalah 3200 tweet.

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

# In[ ]:




