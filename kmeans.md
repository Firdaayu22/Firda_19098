# Algoritma *Clustering*

## Pengertian *Clustering*

*Clustering* atau klasterisasi adalah metode pengelompokan data. Menurut Tan, 2006 clustering merupakan  proses pengelompokan data menjadi beberapa *cluster* atau kelompok sehingga kesamaan data dalam  *cluster* dimaksimalkan dan kesamaan data antar *cluster* diminimalkan. 

## Jenis-Jenis Algoritma Clustering

- #### Centeroid-Based Clustering

  Metode berbasis centeroid merupakan salah satu metode clustering yang paling banyak digunakan. Mengelompokkan data  ke dalam *non-hierarchical clusters*. *Non-hierarchical clusters* adalah tipe cluster efisien yang sensitif terhadap *outlier*. Tipe ini juga termasuk dalam algoritma iteratif, dimana setiap *cluster* biasanya  dibentuk terlebih dahulu dari jarak terdekat dengan *centroid* (pusat cluster). Kemudian data yang sama dikelompokkan sebanyak mungkin. Oleh karena itu, pengelompokan jenis ini telah menjadi metode yang sangat umum [1].

- #### Density-Based Clustering

  Metode ini bekerja dengan menggabungkan data yang sama dengan data yang sama lainnya. Metode ini biasanya digunakan untuk mengelompokkan data yang berbeda berdasarkan nilai dimensi yang tinggi. Kemudian *cluster* dibentuk oleh sumber data yang sama. Data dengan kepadatan yang sama dengan jumlah tertentu dapat disebut grup. Sebaliknya, sejumlah kecil data yang sama disebut *outlier* atau *noise* [1].

- #### Distribution-Based Clustering

  Pengelompokan atau asumsi data dilakukan dengan metode distribusi. Oleh karena itu, metode yang satu ini disebut *Distribution Based*. Peningkatan jarak dari pusat cluster diharapkan dapat mengurangi jumlah data semi-identik. Pengelompokan data biner dapat digunakan untuk mengelompokkan data besar atau kecil. Cara ini juga banyak digunakan karena hasilnya padat dan tingkat identiknya cukup tinggi [1].

- #### Hierarchical Clustering

  *Hierarchical Clustering* atau *Connectivity Based Cluster*, penggunaannya  mirip dengan cluster berbasis *Centroid*. Sederhananya, metode ini mengidentifikasi data berdasarkan jarak terdekat pada kondisi diskriminan tertentu. Selain itu, metode ini bekerja pada sistem dasar di mana data terdekat memiliki sifat yang sama dibandingkan dengan data yang jauh. Dendogram merupakan dasar yang digunakan untuk merepresentasikan pengelompokan data [1].

## Step *Clustering*

Tahapan atau langkah-langkah dari *Clustering* adalah sebagai berikut:

## 1. Persiapan Data

#### Import modul

*Import* modul-modul yang digunakan seperti *pandas, numpy, nltk, sklearn* dan lain sebagainya.

```python
import pandas as pd 
import numpy as np
import string 
import re #regex library
import nltk
import swifter

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
```



#### Read data

Setelah *import modul, read data* yang telah di-*crawling* menggunakan *library pandas*.

```python
Tf_data = pd.read_csv("datapta.csv")
Tf_data
```



## 2. *Text Pre-processing*

### Pengertian *Pre-processing*

Pra-pemrosesan teks merupakan proses yang dilakukan untuk mengolah data yang akan dianalisis menjadi data yang lebih mudah dipahami seperti menghapus noise, missing value, dan data yang tidak konsisten. Adapun tahapan dari *text pre-processing* adalah sebagai berikut:

### *Step Pre-processing*

#### Case Folding

Case folding merupakan proses untuk mengubah kalimat dalam teks menjadi huruf kecil [2]. Adapun *code*nya adalah sebagai berikut:

```python
Tf_data['TP_Abstraksi'] = Tf_data['abstraksi'].str.lower()

print(Tf_data['TP_Abstraksi'].head(5))
```



#### Tokenizing

Tokenizing merupakan proses untuk membersihkan dan menghapus text spesial seperti simbol, angka, link, whitespace dan lain-lain. Adapun *code*nya adalah sebagai berikut:

```python
#Hapus teks spesial
def remove_text_special(text):
    # hapus tab, new line, dan back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # hapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # hapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # hapus URL
    return text.replace("http://", " ").replace("https://", " ")

#hapus angka
def remove_number(text):
    return  re.sub(r"\d+", "", text)

#hapus tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

#hapus whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

#hapus multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

#hapus single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

#tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

#menerapkan fungsi-fungsi yang telah dibuat diatas
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_text_special)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_number)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_punctuation)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_whitespace_LT)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_whitespace_multiple)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_singl_char)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(word_tokenize_wrapper)

print(Tf_data['TP_Abstraksi'].head())
```



#### Stopword Removal

Stopword removal merupakan proses untuk menghapus kata yang tidak penting dan tidak memiliki makna seperti "yang", "dari" dan lain-lain [2]. Adapun *code*nya adalah sebagai berikut:

```python
stop_words = set(stopwords.words('indonesian'))

def stopwords_remove(words):
    return [word for word in words if not word in stop_words]
    
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(stopwords_remove)

print(Tf_data['TP_Abstraksi'].head())
```



#### Stemming

Stemming merupakan proses untuk mendapatkan kata dasar dari suatu kata atau term. Adapun kodennya adalah sebagai berikut:

```python
#membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stemming
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in Tf_data['TP_Abstraksi']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])

print("------------------------")

#terapkan stemmed term ke dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].swifter.apply(get_stemmed_term)
print(Tf_data['TP_Abstraksi'])
```



## 3. Term Weighting

*Term frequency-inverse document frequency* (TF-IDF) adalah metode yang digunakan untuk menghitung berat setiap term dengan mencari seberapa jauh hubungan antara kata atau istilah dengan dokumen. Metode TF-IDF adalah efisien dan memiliki hasil yang akurat. Proses yang dilakukan di metode ini untuk menghitung nilai *term frequency* (TF) dan *inverse document frequency* (IDF) untuk setiap token di setiap token di setiap dokumen di korpus [2]. 

Untuk menentukan besaran nilai IDF maka menggunakan rumus:

$$
IDF_{i, j}=log \frac{D}{df_{j}}
$$

Dimana,

D : jumlah semua dokumen.

dfj : jumlah dokumen yang mengandung term (j).

Sehingga persamaan dari term frequency-inverse document frequency adalah sebagai berikut:

$$
w_{i, j}=t f_{i, j} \times {idf_{j}} \\
w_{i, j}=t f_{i, j} \times \log \frac{D}{df_{j}}
$$

Dimana,

Wij : bobot term (j) terhadap (i).

tfij : jumlah kemunculan term (j) dalam dokumen (i).

Adapun *code*nya adalah sebagai berikut:

```python
vectorizer = TfidfVectorizer(stop_words='english')
pta = []
for data in Tf_data['TP_Abstraksi']:
    isi = ''
    for term in data:
        isi += term + ' '
    pta.append(isi)

vectorizer.fit(pta)
vect = vectorizer.fit_transform(pta)
print(vect)
```

- Tf_data['TP_Abstraksi'] merupakan output dari proses text pre-processing

- vect digunakan sebagai input untuk langkah selanjutnya yaitu K-Means.

  

## 4. *K-Means Clustering*

### Pengertian K-Means

K-means merupakan  algoritma pembelajaran yang tidak terawasi. K-Means digunakan untuk mengelompokkan data ke dalam data cluster dan dapat menerima data tanpa ada label kategori. Adapun langkah-langkah dari K-Means adalah sebagai berikut:

### Step K-Means

- **Tentukan Jumlah Cluster (K)**

- **Pilih Titik Acak Sebanyak K**

  Adapun codenya adalah sebagai berikut:

  ```python
  true_k = 5
  ```

  

- **Menghitung jarak data dengan centroid.** 

  Adapun persamaan *Euclidean Distance* yang digunakan adalah sebagai berikut:

$$
d=\sqrt{\left(x_{2}-x_{1}\right)^{2}+\left(y_{2}-y_{1}\right)^{2}}
$$

Adapun codenya adalah sebagai berikut:

```python
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(vect)
```



- **Label Semua Data Berdasarkan Cluster Terdekat**

  Adapun codenya adalah sebagai berikut:

  ```python
  print("Top terms per cluster:")
  order_centroids = model.cluster_centers_.argsort()[:, ::-1]
  terms = vectorizer.get_feature_names()
  for i in range(true_k):
      print("Cluster %d:" % i),
      for ind in order_centroids[i, :10]:
          print('%s' % terms[ind],end=" ")
      print("\n")
  ```

  

- **Gunakan PCA untuk mereduksi dimensi**

  Adapun codenya adalah sebagai berikut:

  ```python
  pca = PCA(n_components=3, random_state=42)
  # pass vector to the pca and store the reduced vectors into pca_vecs
  pca_vecs = pca.fit_transform(vect.toarray())
  x0 = pca_vecs[:, 0]
  x1 = pca_vecs[:, 1]
  ```

  ```python
  #menambahkan cluster ke dataset
  Tf_data['cluster'] = model.labels_
  Tf_data['x0'] = x0
  Tf_data['x1'] = x1
  print (x0)
  print (x1)
  ```

  

- **Label Ulang Data Berdasarkan Jarak Terdekat terhadap Centroid Baru**

- **Ulangi Langkah Sampai Tidak Ada Pergerakan Lagi.**

- **Grafik hasil K-Means**

  Adapun codenya adalah sebagai berikut:

  ```python
  plt.figure(figsize=(12, 7))
  plt.title("Hasil KMeans Clustering", fontdict={"fontsize": 18})
  
  plt.xlabel("X0", fontdict={"fontsize": 16})
  plt.ylabel("X1", fontdict={"fontsize": 16})
  
  # create scatter plot with seaborn, where hue is the class used to group the data
  sns.scatterplot(data=Tf_data, x='x0', y='x1', hue='cluster', palette="viridis")
  plt.show()
  ```

  

### Kelebihan K-Means

- Algoritma yang sederhana dan mudah dipahami
- Cepat pemrosesannya
- Tersedia di berbagai tools atau software
- Mudah dalam penerapannya
- Selalu memberikan hasil, apapun datanya [3].

### Kekurangan K-Means

- Hasilnya sensitif terhadap jumlah cluster (K).
- Cepat pemrosesannya
- Sensitif terhadap pencilan atau outlier.
- Sensitif terhadap data dengan variabel yang memiliki skala berbeda [3]. 

## Referensi

[1] : Definisi Clustering, Manfaat, Jenis, Cara Kerja dan Contoh Penggunaannya. Website: https://www.pengadaanbarang.co.id/2022/03/manfaat-clustering-adalah.html, diakses pada 30 Juni 2022.

[2] : Fitri, H., dkk (2021). Topic Modeling in the News Document on Sustainable. Tersedia dari jurnal.ugm.ac.id [https://www.jurnal.ugm.ac.id/ijitee/article/view/67467/32203]. 

[3] : K-means Clustering: Pengertian, Metode Algoritma, Beserta Contoh. Website: https://geospasialis.com/k-means-clustering/, diakses 30 Juni 2022.
