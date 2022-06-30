# Topic Modelling

## Pengertian *Topic Modelling*

*Topic modelling* merupakan bentuk penambangan teks yang digunakan untuk mengidentifikasi pola dalam korpus. *Topic Modelling* juga merupakan pendekatan  yang cukup handal untuk penambangan teks untuk menemukan data teks tersembunyi dan menemukan hubungan antara satu teks  dengan teks lainnya dari korpus (Jelodar, et al., 2018) atau dapat disimpulkan bahwa *Topic Modelling* merupakan mengelompokkan data tekstual berdasarkan topik tertentu.

## Step *Topic Modelling*

Tahapan atau langkah-langkah dari *Topic Modelling* adalah sebagai berikut:

## 1. *Crawling Data*

### Pengertian *Crawling Data*

Crawling Data merupakan proses melakukan pencarian dan pemindaian sebuah konten/data pada website. Data crawling adalah aktivitas crawler untuk mengindeks sebuah data dan mengunduh dari internet yang kemudian akan disimpan ke dalam database [1].

Proses *crawling* dapat dilakukan dengan berbagai cara namun pada kali ini proses *crawling* dilakukan menggunakan *Scrapy Python*. Adapun untuk tahapan-tahapan dari proses crawling adalah sebagai berikut:

### *Step Crawling Data*

#### Mencari URL yang dituju.

Tentukan URL yang akan diambil datanya, sebagai contoh pada proses kali ini menggambil data tugas akhir Universitas Trunojoyo Madura yakni https://pta.trunojoyo.ac.id.



#### Install scrapy

Setelah memastikan URL yang dituju, maka tahap selanjutnya adalah menginstal scrapy menggunakan perintah :

```python
pip install scrapy
```



#### Membuat project baru

Setelah berhasil menginstal scrapy maka tahap selanjutnya adalah membuat *project scrapy* baru menggunakan perintah:

```python
scrapy startproject projectname
```

Dengan, 

- projectname : nama project baru

  

#### Membuat *file spider* baru

Apabila telah berhasil membuat *project scrapy* baru maka masuk ke direktori folder tersebut dan kemudian membuat *file spider* baru mengunakan perintah:

```python
scrapy genspider newspider urltujuan
```

Dengan, 

- newspider : nama file spider baru.

- urltujuan : link URL yang akan dituju.

  

#### Edit file spider

Buka file spider.py pada *text editor*, buka terminal dan buka *scrapy console* menggunakan perintah:

```python
scrapy shell ‘pta.tunojoyo.ac.id’
```

Setelah *console* berhasil terbuka, lakukan *inspect element* pada *website* tersebut untuk mendapatkan *element* yang berisi data-data yang ingin dicari seperti judul, penulis, pembimbing I, pembimbing II dan abstrak dan jalankan pada *console* apabila telah berhasil maka *copy paste code* tadi untuk di-*input* ke dalam *file spider* tadi, sehingga hasilnya adalah sebagai berikut:

```python
import scrapy

class WebminingSpider(scrapy.Spider):
    name = 'webmining'
    allowed_domains = ['pta.trunojoyo.ac.id']
    start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/7/'+str(x)+'' for x in range(1,5)]
    
	#element button yang mengarah ke link data yang dicari 
    def parse(self, response):
        for link in response.css('a.gray.button::attr(href)'):
            yield response.follow(link.get(), callback=self.parse_categories)
	
    #element tiap data yang dicari
    def parse_categories(self, response):
        product = response.css('div#content_journal ul li')
        for link in product:
            yield {
                #mendapatkan data "judul"
                #strip() digunakan untuk merapikan data
                'judul' : product.css('div a.title::text').get().strip(),
                #mendapatkan data "penulis"
                'penulis' : product.css('div div:nth-child(2) span::text').get().strip(),
                #mendapatkan data "abstraksi"
                'abstraksi' : product.css('div div:nth-child(2) p::text').get().strip(),
            }
```

Pada *"start_urls"* link dapat diubah-ubah sesuai kebutuhan dan pada kali ini dilakukan perulangan atau iterasi sebanyak 5 yang berfokus pada tugas akhir mahasiswa Manajemen.

#### Simpan data hasil crawling

Setelah berhasil mengekstrak data maka tahap selanjutnya adalah menyimpan data sesuai format yang diinginkan menggunakan perintah:

```python
scrapy crawl newspider -o hasil.csv
```

Dengan,

- newspider : file spider yang telah dibuat.
- hasil.csv : file csv untuk menyimpan hasil *crawling*.

Buka kembali console dan ketikkan perintah diatas dan tunggu hingga proses selesai dan data siap digunakan.



## 2. Persiapan Data

#### Import modul

*Import* modul-modul yang digunakan seperti *pandas, numpy, nltk, sklearn* dan lain sebagainya.

```python
import pandas as pd 
import numpy as np
import string 
import re #regex library
 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
```



#### Read data

Setelah *import modul, read data* yang telah di-*crawling* menggunakan *library pandas*.

```python
Tf_data = pd.read_csv("datapta.csv")
Tf_data
```



## 3. *Text Pre-processing*

### Pengertian *Pre-processing*

Pra-pemrosesan teks merupakan proses yang dilakukan untuk mengolah data yang akan dianalisis menjadi data yang lebih mudah dipahami seperti menghapus noise, missing value, dan data yang tidak konsisten. Adapun tahapan dari *text pre-processing* adalah sebagai berikut:

### *Step Pre-processing*

#### Case Folding

Case folding merupakan proses untuk mengubah kalimat dalam teks menjadi huruf kecil [2]. Adapun *code*nya adalah sebagai berikut:

```python
Tf_data['TP_Abstraksi'] = Tf_data['abstraksi'].str.lower()

print(Tf_data['TP_Abstraksi'].head(5))
```



#### Cleaning Data

Cleaning data merupakan proses untuk membersihkan dan menghapus text spesial seperti simbol, angka, link, whitespace dan lain-lain. Adapun *code*nya adalah sebagai berikut:

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

#menerapkan fungsi-fungsi yang telah dibuat diatas
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_text_special)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_number)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_punctuation)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_whitespace_LT)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_whitespace_multiple)
Tf_data['TP_Abstraksi'] = Tf_data['TP_Abstraksi'].apply(remove_singl_char)

print(Tf_data['TP_Abstraksi'].head())
```



#### Stopword Removal

Stopword removal merupakan proses untuk menghapus kata yang tidak penting dan tidak memiliki makna seperti "yang", "dari" dan lain-lain [2]. Adapun *code*nya adalah sebagai berikut:

```python
stop_words = stopwords.words('indonesian')
stop_words.extend(['baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya'])
stop_words = set(stop_words)
```



## 4. Term Weighting

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

Berapapun besarnya nilai *tfij*, apabila *D = dfj*, maka hasilnya adalah 0 (nol), dikarenakan hasil dari *log 1* adalah 0. Untuk itu dapat ditambahkan nilai 1 pada sisi IDF, sehingga perhitungan bobotnya menjadi sebagai berikut :

$$
w_{i, j}=t f_{i, j} \times \log \frac{D}{df_{j}} + 1
$$

Adapun *code*nya adalah sebagai berikut:

```python
vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)
vect_text=vect.fit_transform(Tf_data['TP_Abstraksi'])

print(vect_text)
```

- Tf_data['TP_Abstrak'] merupakan output dari proses text pre-processing

- vect_text digunakan sebagai input untuk langkah selanjutnya yaitu LSA.

  

## 5. *Latent Semantic Analysis* (LSA)

### Pengertian LSA

*Latent Semantic Analysis* (LSA) adalah metode analisis struktur semantik teks menggunakan model statistik matematis. LSA dapat digunakan untuk menilai esai dengan mengubahnya menjadi matriks yang diberi skor untuk setiap term dan mencari kesamaan dengan term referensi [3]. Adapun langkah-langkah dari LSA adalah sebagai berikut:

### *Step* LSA

#### Singular Value Decomposition

Setelah melalui proses *text pre-processing* dan *term weighting* maka proses selanjutnya adalah proses *singular value decomposition.* *Singular Value Decomposition* (SVD) merupakan  teknik reduksi dimensi yang membantu mengurangi nilai kompleksitas saat pemrosesan *term-document matrix*. SVD adalah teorema aljabar linier yang menunjukkan bahwa persegi panjang dalam *term-document matrix* dapat dipecah / didekomposisi menjadi tiga matriks, yakni :

- Matriks ortogonal  U
- Matriks diagonal D
- Transpos matriks ortogonal V

Sehingga didapatkan persamaan:

$$
\begin{array}{ll}
A_{m n} & =U_{m m} x D_{m n} x V_{n n}^{T} \\
\mathrm{~A}_{m n} & =\quad \text { matriks awal } \\
\mathrm{U}_{m m} & =\quad \text { matriks ortogonal U } \\
\mathrm{D}_{m n}= & \text { matriks diagonal D } \\
\mathrm{V}_{n n}^{\top}= & \text { transpose matriks ortogonal V }
\end{array}
$$

Adapun codenya adalah sebagai berikut:

```python
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_topik=lsa_model.fit_transform(vect_text)
print(lsa_topik)
```

Hasil dari proses SVD adalah vektor yang  digunakan untuk menghitung kemiripan dokumen.



#### Menampilkan hasil topik tiap dokumen.

Setelah proses SVD maka dapat dilakukan proses menampilkan hasil topik tiap dokumen, dan pada kali ini contoh topik yang ditampilkan adalah topik pada dokumen 1 sebagai berikut:

```python
l=lsa_topik[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i+1," : ",topic*100)
```



#### Menghasilkan nilai komponen tiap topik.

Menampilkan nilai komponen tiap topik juga penting dilakukan karena sebagai input untuk proses selanjutnya yakni menampilkan kata penting pada setiap topik.

```python
print(lsa_model.components_.shape) #jml topik, jml kata
print(lsa_model.components_)
```



#### Menghasilkan kata penting tiap topik.

Setelah didapatkan nilai komponen maka selanjutnya dapat digunakan untuk menghasilkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan. Adapun *code*nya adalah sebagai berikut:

```python
# most important words for each topic
# mendapatkan term
vocab = vect.get_feature_names_out()

print("Kata Penting :\n")
for i, komponen in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, komponen)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
```

Adapun hasil akhir dari LSA adalah sebagai berikut:

| Kata Penting :                                               |
| :----------------------------------------------------------- |
| Topic 0:  variabel penelitian konsumen harga uji pengaruh kepuasan pembelian keputusan signifikan |
| Topic 1:  kerja pegawai prestasi pengembangan kepemimpinan tipe produktivitas kompensasi dinas kabupaten |
| Topic 2:  ratio lamongan bank npl resiko risk bersaing inovasi keunggulan tingkat |
| Topic 3:  kepuasan akademik portal pelanggan langsung kualitas jaminan indeks madura universitas |
| Topic 4:  dimensi persepsi minat banking internet association brand cranberries distro kiddrock |
| Topic 5:  kompetensi langsung dosen kinerja pembelian kompensasi hipotesis honda vario pedagogik |
| Topic 6:  biaya produk bersaing inovasi keunggulan cacat pelaporan profitabilitas langsung risiko |
| Topic 7:  biaya variabel cacat pelaporan profitabilitas kualitas produk risiko kompensasi kompetensi |
| Topic 8:  kompetensi bersaing inovasi keunggulan kinerja dosen optik reza pemasaran harga |
| Topic 9:  kompetensi dosen uji biaya akademik cacat pelaporan profitabilitas pln risiko |



## Referensi

[1] : Crawling adalah ? Bagaimana Cara Kerja Web Crawler?. Website: http://maximadigital.id/crawling-adalah/, diakses pada 16 Juni 2022.

[2] : Fitri, H., dkk (2021). Topic Modeling in the News Document on Sustainable. Tersedia dari jurnal.ugm.ac.id [https://www.jurnal.ugm.ac.id/ijitee/article/view/67467/32203]. 

[3] : Penggunaan Latent Semantic Analysis (LSA) dalam Pemrosesan Teks. Website: https://socs.binus.ac.id/2015/08/03/penggunaan-latent-semantic-analysis-lsa-dalam-pemrosesan-teks/, diakses 16 Juni 2022.
