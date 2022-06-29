# *Latent Semantic Analysis* (LSA)

### Pengertian LSA

*Latent Semantic Analysis* (LSA) adalah metode analisis struktur semantik teks menggunakan model statistik matematis. LSA dapat digunakan untuk menilai esai dengan mengubahnya menjadi matriks yang diberi skor untuk setiap term dan mencari kesamaan dengan term referensi [1]. Adapun langkah-langkah dari LSA adalah sebagai berikut:

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

Adapun contoh perhitungan dengan SVD adalah sebagai berikut:

$$
MatriksA=\begin{bmatrix} 1 & 2 & -1\\ 2 & 1 & -1\end{bmatrix}
$$

$$
AA^{T}=\begin{bmatrix} 1 & 2 & -1\\ 2 & 1 & -1\end{bmatrix} \begin{bmatrix} 1 & 2\\ 2 & 1 \\-1 & -1\end{bmatrix}
$$

$$
AA^{T} =\begin{bmatrix} 6 & 5\\ 5 & 6\end{bmatrix}
$$

Eigenvalue:

$$
det(A - \lambda) = 0\\
det\begin{pmatrix} 6-\lambda & 5 \\ 5 & 6-\lambda \end{pmatrix} = 0 \\
(6-\lambda)^2 - 25 = 0\\
\lambda_1 = 11 \\
\lambda_2 = 1
$$

Eigenvector:

$$
(A - \lambda) x= 0\\
\begin{pmatrix} 6-\lambda & 5 \\ 5 & 6-\lambda \end{pmatrix}\begin{pmatrix} x1\\ x2\end{pmatrix} = 0 \\
(6-\lambda)x1 + 5x2 = 0\\
5x1 + (6-\lambda)x2 = 0
$$

Jika:

$$
\lambda = 11\\
(6-11)x1 + 5x2 = 0\\
5x1 + (6-11)x2 = 0\\
x1 = x2\\
x1 = -1 \\
x2 = -1
$$

Jika:

$$
\lambda = 1\\
(6-1)x1 + 5x2 = 0\\
5x1 + (6-1)x2 = 0\\
x1 = -x2\\
x1 = -1 \\
x2 = 1
$$

$$
Du = \begin{bmatrix} \sqrt{11}& 0\\ 0 & \sqrt{1}\end{bmatrix} \\
$$

Normalisasi orthonormal dengan menjadikan unit vektor:

$$
u = \begin{bmatrix} -1 & -1\\ -1 & 1\end{bmatrix}\\
u = \begin{bmatrix} \frac {-1}{\sqrt{2}}& \frac {-1}{\sqrt{2}}\\ \frac {-1}{\sqrt{2}} & \frac {1}{\sqrt{2}}\end{bmatrix}\\
u = \begin{bmatrix} -0,71 & -0,71\\ -0,71 & 0,71\end{bmatrix}
$$

$$
A^{T}A = \begin{bmatrix} 1 & 2\\ 2 & 1\\-1 & -1\end{bmatrix} \begin{bmatrix} 1 & 2 & -1\\ 2 & 1 & -1\end{bmatrix}\\
$$

$$
A^{T}A = \begin{bmatrix} 5 & 4 & -3\\ 4 & 5 & -3\\-3 & -3 & 2\end{bmatrix}
$$

$$
V = \begin{bmatrix} -0,64 & 0,71 & 0,30\\ -0,64 & -0,71 & 0,30\\0,43 & 0 & 0,91\end{bmatrix}
$$

$$
SVD(A) = UxDxV^{T}
$$

$$
SVD(A) = \begin{bmatrix} -0,71 & -0,71\\ -0,71 & 0,71\end{bmatrix} \begin{bmatrix} 3,32& 0 & 0\\ 0 & 1 & 0\end{bmatrix} \begin{bmatrix} -0,64 & 0,71 & 0,30\\ -0,64 & -0,71 & 0,30\\0,43 & 0 & 0,91\end{bmatrix}
$$

$$
SVD(A) = \begin{bmatrix} -0,71 & -0,71\\ -0,71 & 0,71\end{bmatrix} \begin{bmatrix} 3,32& 0 & 0\\ 0 & 1 & 0\end{bmatrix} \begin{bmatrix} -0,64 & -0,64 & 0,43\\ 0,71 & -0,71 & 0\\0,30 & -0,30 & 0,91\end{bmatrix}
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
| Topic 0:  citra batik metode data proses sistem nilai hasil tekstur pengenalan |
| Topic 1:  citra batik tekstur ciri fitur kemiripan ekstraksi perolehan isi gambar |
| Topic 2:  bahasa algoritma madura mobile android teknologi pembelajaran aplikasi arsitektur pencarian |
| Topic 3:  tangan tulisan pengenalan sidik jari telapak skenario proses carakan senyum |
| Topic 4:  produksi peramalan perusahaan penjualan algoritma permintaan pelanggan penjadwalan bahasa komputer |
| Topic 5:  arsitektur bangkalan informasi dinas pelayanan bisnis tahapan peramalan kepegawaian sistem |
| Topic 6:  sidik jari pendeteksian citra manusia skenario gizi region titik pasien |
| Topic 7:  gizi pasien status peramalan obat balita penentuan data nilai kebutuhan |
| Topic 8:  gizi mobile pasien citra status android teknologi balita gerakan perusahaan |
| Topic 9:  gizi bahasa madura pasien pelanggan status batik perusahaan sidik jari |



## Referensi

[1] : Penggunaan Latent Semantic Analysis (LSA) dalam Pemrosesan Teks. Website: https://socs.binus.ac.id/2015/08/03/penggunaan-latent-semantic-analysis-lsa-dalam-pemrosesan-teks/, diakses 16 Juni 2022.
