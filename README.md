# Predictive Analytics: Red Wine Quality Study Case

## Daftar Isi
- [Predictive Analytics: Red Wine Quality Study Case](#pedictive-analytics-red-wine-quality-study-case)
  - [Daftar Isi](#daftar-isi)
  - [Problem Domain](#problem-domain)
  - [Business Understanding](#business-understanding)
    - [Problem Statement](#problem-statement)
    - [Goals](#goals)
    - [Solution Statement](#solution-statement)
  - [Data Understanding](#data-understanding)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Univariate Analysis](#univariate-analysis)
    - [Multi-Variate Analysis](#multi-variate-analysis)
  - [Data Preparation](#data-preparation)
  - [Modeling](#modeling)
  - [Evaluation](#evaluation)
    - [Cross Validation](#cross-validation)
    - [Confusion Matrix](#confusion-matrix)
    - [Classification Report](#classification-report)


## Problem Domain
Saat ini banyak *wine* beredar di pasaran, sehingga penilaian terhadap *wine* mutlak diperlukan, karena hal ini memberikan pengaruh yang sangat besar terhadap keputusan pembelian konsumen.

Dalam hal ini, kita akan memprediksi kualitas dari sebuah *wine* berdasarkan beberapa atribut yang ada. Sehingga, kita dapat mengetahui apakah *wine* tersebut memiliki kualitas yang baik atau tidak.

Berdasarkan permasalahan tersebut, penulis ingin membuat sebuah model *machine learning* yang dapat memprediksi kualitas dari sebuah *wine* sehingga dapat memberikan informasi kepada konsumen apakah *wine* tersebut layak untuk dibeli atau tidak.



## Business Understanding
### Problem Statement
Berdasarkan *problem domain* di atas, maka diperoleh *problem statement* sebagai berikut:
- Apa saja yang perlu dilakukan dalam *data preparation* untuk membuat model *machine learning* yang baik?
- Bagaimana cara memilih model *machine learning* yang tepat untuk memprediksi kualitas dari sebuah *wine*?

### Goals
Berdasarkan *problem statement* di atas, maka diperoleh *goals* sebagai berikut:
- Untuk melakukan tahap *data preparation*, agar dapat memperoleh data yang bersih dan dapat digunakan untuk membuat model *machine learning*.
- Untuk memilih model *machine learning* yang tepat, agar dapat memperoleh model yang baik dan dapat digunakan untuk memprediksi kualitas dari sebuah *wine*.

### Solution Statement
Berdasarkan *goals* di atas, maka diperoleh beberapa *solution statement* untuk menyelesaikan masalah tersebut, yaitu:
- Melakukan *data preparation* untuk memastikan data siap digunakan dalam membuat model *machine learning*. Beberapa tahapan yang dapat dilakukan, antara lain *handling missing value*, *handling outliers*, *feature engineering*, *splitting data*, serta *standarisasi*.
- Cara memilih model *machine learning* yang tepat dilakukan dengan beberapa tahapan, yaitu:
  - Menggunakan *LazyPredict* untuk membandingkan beberapa model *machine learning* yang dapat digunakan.
  - Memilih 5 model *machine learning* yang memiliki nilai *accuracy* tertinggi.
  - Mencoba membangun model *Hard Voting Classifier* dengan menggunakan 5 model *machine learning* yang telah dipilih.
  - Mengevaluasi model tersebut dengan menggunakan *cross validation*, *confusion matrix*, dan *classification report*.



## Data Understanding
Data yang diguanakan dalam kasus ini adalah data *wine* yang dapat diunduh dari [sini](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009). Data tersebut berisi 1599 baris dan 12 kolom, dengan kolom-kolom tersebut adalah sebagai berikut:
- **fixed acidity**: kandungan asam yang tidak dapat diubah oleh bakteri (*fixed*).
- **volatile acidity**: kandungan asam yang dapat diubah oleh bakteri (*volatile*).
- **citric acid**: kandungan asam sitrat.
- **residual sugar**: kandungan gula yang tersisa setelah fermentasi.
- **chlorides**: kandungan klorida.
- **free sulfur dioxide**: kandungan sulfur dioksida yang bebas.
- **total sulfur dioxide**: kandungan sulfur dioksida yang total.
- **density**: kandungan massa jenis.
- **pH**: kandungan pH.
- **sulphates**: kandungan sulfat.
- **alcohol**: kandungan alkohol.
- **label**: kualitas dari *wine* tersebut.

> Note: pada studi kasus ini, penulis telah mengubah target yang awalnya multi-class menjadi binary-class. *Wine* yang memiliki kualitas lebih dari 5 akan dianggap memiliki kualitas yang baik, sedangkan *wine* yang memiliki kualitas kurang dari sama dengan 5 akan dianggap memiliki kualitas yang buruk.

### Descriptive Statistics
Tabel 1. Descriptive Statistics dari Data *Wine*
|       |   fixed.acidity |   volatile.acidity |   citric.acid |   residual.sugar |    chlorides |   free.sulfur.dioxide |   total.sulfur.dioxide |       density |          pH |   sulphates |    alcohol |       label |
|:------|----------------:|-------------------:|--------------:|-----------------:|-------------:|----------------------:|-----------------------:|--------------:|------------:|------------:|-----------:|------------:|
| count |      1599       |        1599        |   1599        |       1599       | 1599         |             1599      |              1599      | 1599          | 1599        | 1599        | 1599       | 1599        |
| mean  |         8.31964 |           0.527821 |      0.270976 |          2.53881 |    0.0874665 |               15.8749 |                46.4678 |    0.996747   |    3.31111  |    0.658149 |   10.423   |    0.534709 |
| std   |         1.7411  |           0.17906  |      0.194801 |          1.40993 |    0.0470653 |               10.4602 |                32.8953 |    0.00188733 |    0.154386 |    0.169507 |    1.06567 |    0.49895  |
| min   |         4.6     |           0.12     |      0        |          0.9     |    0.012     |                1      |                 6      |    0.99007    |    2.74     |    0.33     |    8.4     |    0        |
| 25%   |         7.1     |           0.39     |      0.09     |          1.9     |    0.07      |                7      |                22      |    0.9956     |    3.21     |    0.55     |    9.5     |    0        |
| 50%   |         7.9     |           0.52     |      0.26     |          2.2     |    0.079     |               14      |                38      |    0.99675    |    3.31     |    0.62     |   10.2     |    1        |
| 75%   |         9.2     |           0.64     |      0.42     |          2.6     |    0.09      |               21      |                62      |    0.997835   |    3.4      |    0.73     |   11.1     |    1        |
| max   |        15.9     |           1.58     |      1        |         15.5     |    0.611     |               72      |               289      |    1.00369    |    4.01     |    2        |   14.9     |    1        |

Beberapa informasi yang dapat diperoleh dari tabel di atas adalah sebagai berikut:
- Datasets ini terdiri dari 1599 baris dan 12 kolom.
- Beberapa kolom memiliki nilai *skewness* yang cukup tinggi. Hal ini dapat dilihat dari nilai *mean* dan *median* yang berbeda cukup jauh.
- Beberapa kolom juga terindikasi memiliki *outlier*. Terlihat bahwa pada nilai *mean* yang cukup kecil jika dibandingkan dengan nilai *max*.
- Tidak ada missing value pada datasets ini.

### Univariate Analysis
![Countplot dari Kolom Label](/img/01a_count_label.png)

Gambar 1. Countplot dari Kolom Label
</br>
</br>

![Histogram dari Kolom Fitur-Fitur](/img/01b_hist_feature.png)

Gambar 2. Histogram dari Kolom Fitur-Fitur
</br>
</br>

![Boxplot dari Kolom Fitur-Fitur](/img/01c_boxplot.png)

Gambar 3. Boxplot dari Kolom Fitur-Fitur
</br>
</br>

Beberapa informasi yang dapat diperoleh dari beberapa gambar di atas adalah sebagai berikut:
- Dari gambar 1, dapat dilihat bahwa jumlah *wine* yang memiliki kualitas baik lebih banyak dibandingkan dengan *wine* yang memiliki kualitas buruk, **perbandingan tersebut sekitar 53:47** (dapat dikatakan masih *balanced*).
- Terdapat *outlier* pada beberapa kolom, sehingga perlu dilakukan ***handling outlier*** pada tahap *data preparation*.

### Multi-Variate Analysis
![Heatmap dari Korelasi Fitur-Fitur](/img/02_heatmap.png)

Gambar 4. Heatmap dari Korelasi Fitur-Fitur

Beberapa informasi yang dapat diperoleh dari gambar di atas adalah sebagai berikut:
- Terdapat beberapa fitur yang memiliki korelasi yang cukup tinggi, yaitu:
    - **fixed.acidity** dan **citric.acid** (0.67)
    - **fixed.acidity** dan **density** (0.67)
    - **citric.acid** dan **density** (0.54)
    - **density** dan **pH** (-0.69)
    - **density** dan **alcohol** (-0.50)
    - **pH** dan **alcohol** (0.12)



## Data Preparation
Berikut adalah beberapa teknik yang digunakan untuk melakukan *data preparation* pada dataset ini:
- **Handling Outlier**. Terdapat beberapa fitur yang memiliki *outlier*. Untuk mengatasi hal ini, digunakan teknik ***interquartile range (IQR)*** dengan rumus sebagai berikut:
    - Q1 = 25% dari data
    - Q3 = 75% dari data
    - IQR = Q3 - Q1
    - Lower Bound = Q1 - 1.5 * IQR
    - Upper Bound = Q3 + 1.5 * IQR
    - Data yang berada di luar Lower Bound dan Upper Bound dianggap sebagai *outlier*. Data *outlier* tersebut akan dihapus.
- **Data Transformation**. Setelah dilakukan *handling outlier*, terdapat beberapa fitur yang memiliki *skewness* yang cukup tinggi. Untuk mengatasi hal ini, digunakan teknik ***log transformation***. Sehingga, nilai *skewness* dari fitur-fitur tersebut akan berkurang dan distribusi dari fitur-fitur tersebut akan lebih normal.
- **Splitting Data**. Data akan dibagi menjadi 2 bagian, yaitu:
    - **Training Data**. Data yang digunakan untuk melakukan *training* model.
    - **Testing Data**. Data yang digunakan untuk melakukan *testing* model.
- **Feature Scaling**. Data yang digunakan untuk *training* model akan di-*scale* menggunakan teknik ***standardization***. Dengan demikian, nilai mean dari setiap fitur akan menjadi 0 dan nilai standar deviasi dari setiap fitur akan menjadi 1 serta distribusi dari setiap fitur akan lebih normal.



## Modeling
Pada tahap ini, akan dilakukan *training* model menggunakan beberapa algoritma *machine learning* menggunakan metode **LazyPredict**. Kemudian akan dipilih 5 model yang memiliki performa terbaik. 

Tabel 2. Hasil LazyPredict
| Model                         |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|:------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
| ExtraTreesClassifier          |   0.810585 |            0.809935 |  0.809935 |   0.810585 |    0.687828  |
| RandomForestClassifier        |   0.788301 |            0.786266 |  0.786266 |   0.787917 |    0.681378  |
| BaggingClassifier             |   0.777159 |            0.77901  |  0.77901  |   0.777297 |    0.169581  |
| RidgeClassifierCV             |   0.774373 |            0.775397 |  0.775397 |   0.774566 |    0.0470045 |
| LogisticRegression            |   0.774373 |            0.77507  |  0.77507  |   0.774563 |    0.0568521 |
| LGBMClassifier                |   0.777159 |            0.774432 |  0.774432 |   0.776468 |    0.375548  |
| LinearDiscriminantAnalysis    |   0.771588 |            0.772438 |  0.772438 |   0.771783 |    0.0886536 |
| RidgeClassifier               |   0.771588 |            0.772438 |  0.772438 |   0.771783 |    0.0372763 |
| LinearSVC                     |   0.771588 |            0.772111 |  0.772111 |   0.771772 |    0.133714  |
| CalibratedClassifierCV        |   0.771588 |            0.772111 |  0.772111 |   0.771772 |    0.402183  |
| QuadraticDiscriminantAnalysis |   0.766017 |            0.767829 |  0.767829 |   0.766162 |    0.0480917 |
| SVC                           |   0.760446 |            0.762566 |  0.762566 |   0.760546 |    0.192991  |
| XGBClassifier                 |   0.75766  |            0.755357 |  0.755357 |   0.757149 |    0.298702  |
| BernoulliNB                   |   0.754875 |            0.75436  |  0.75436  |   0.754947 |    0.0364227 |
| NuSVC                         |   0.749304 |            0.749751 |  0.749751 |   0.749506 |    0.237982  |
| NearestCentroid               |   0.738162 |            0.741841 |  0.741841 |   0.737763 |    0.030174  |
| GaussianNB                    |   0.738162 |            0.741187 |  0.741187 |   0.738015 |    0.0373724 |
| KNeighborsClassifier          |   0.735376 |            0.732342 |  0.732342 |   0.73446  |    0.104009  |
| SGDClassifier                 |   0.732591 |            0.731345 |  0.731345 |   0.732495 |    0.0482693 |
| LabelSpreading                |   0.729805 |            0.727733 |  0.727733 |   0.729392 |    0.139891  |
| LabelPropagation              |   0.729805 |            0.727733 |  0.727733 |   0.729392 |    0.115156  |
| AdaBoostClassifier            |   0.721448 |            0.719838 |  0.719838 |   0.721231 |    0.40655   |
| DecisionTreeClassifier        |   0.696379 |            0.695173 |  0.695173 |   0.696327 |    0.043282  |
| ExtraTreeClassifier           |   0.690808 |            0.689256 |  0.689256 |   0.690634 |    0.040262  |
| Perceptron                    |   0.651811 |            0.644239 |  0.644239 |   0.645307 |    0.0415661 |
| PassiveAggressiveClassifier   |   0.590529 |            0.578823 |  0.578823 |   0.572742 |    0.0419977 |
| DummyClassifier               |   0.529248 |            0.5      |  0.5      |   0.366328 |    0.031126  |

Berikut adalah 5 model yang cukup baik untuk digunakan pada dataset ini.
1. **ExtraTreesClassifier**
2. **RandomForestClassifier**
3. **LogisticRegression**
4. **LinearSVC**
5. **CalibratedClassifierCV**

Berikut adalah hasil akurasi dari *tra ining* 5 model di atas.

Tabel 3. Hasil Akurasi 5 Model Terbaik
| Model                         |   Train Accuracy  |   Test Accuracy |
|:------------------------------|------------------:|----------------:|
| ExtraTreesClassifier          |              1.0  |           0.81  |
| RandomForestClassifier        |              1.0  |           0.79  |
| LogisticRegression            |              0.74 |           0.78  |
| LinearSVC                     |              0.74 |           0.77  |
| CalibratedClassifierCV        |              0.74 |           0.77  |

Jika dilihat secara sekilas, model **ExtraTreesClassifier** memiliki akurasi yang lebih baik dibandingkan model lainnya. Model **ExtraTreesClassifier** memiliki *overfitting* yang cukup tinggi. Hal ini dapat dilihat dari nilai akurasi *training* yang sangat tinggi (1.0) dibandingkan nilai akurasi *testing* yang lebih rendah (0.81). 

Maka dari itu, penulis akan mencoba membangun model lagi dengan menggunakan Hard Voting Classifier. Model ini akan menggunakan 5 model di atas sebagai *base model*. Tujuan dibangunan model ini adalah untuk melihat apakah model ini memiliki akurasi yang lebih baik dibandingkan model **ExtraTreesClassifier**.

Tabel 4. Hasil Akurasi Hard Voting Classifier
| Model                         |   Train Accuracy  |   Test Accuracy |
|:------------------------------|------------------:|----------------:|
| HardVotingClassifier          |              0.75 |           0.78  |
 
Hasil akurasi dari model ini tidak jauh berbeda dengan model **ExtraTreesClassifier** dan model lainnya. Maka dari itu, penulis akan mencoba melakukan *evaluation* terhadap semua model dengan tujuan tidak hanya melihat akurasi saja, tetapi juga melihat sisi lain dan/atau parameter lain dari model yang telah dibangun.

## Evaluation
Model yang dibangun pada studi kasus ini adalah model berjenis klasifikasi. Oleh karena itu, proses evaluasi yang akan dilakukan adalah dengan menggunakan uji ***cross validation***, ***confusion matrix***, dan ***classification report***. 

### Cross Validation
*Cross-validation* (CV) adalah metode statistik yang dapat digunakan untuk mengevaluasi kinerja model atau algoritma dimana data dipisahkan menjadi dua subset yaitu *training data* dan *validation data*. Model atau algoritma dilatih oleh subset *training* dan divalidasi oleh subset *validation*.

*Cross validation* digunakan untuk menghindari *overfitting* dan *underfitting* pada model. *Cross validation* juga dapat digunakan untuk memilih model yang paling baik untuk digunakan pada dataset yang diberikan.

Tabel 5. Hasil Cross Validation
|              |   Score Train |   Score Test |   CV Mean |    CV Std |
|:-------------|--------------:|-------------:|----------:|----------:|
| ExtraTreesClassifier          |      1        |     0.81337  |  0.796472 | 0.0439898 |
| RandomForestClassifier          |      1        |     0.791086 |  0.782014 | 0.0655769 |
| LogisticRegression |      0.741317 |     0.779944 |  0.736489 | 0.0479757 |
| LinearSVC         |      0.74012  |     0.771588 |  0.734079 | 0.038497  |
| CalibratedClassifierCV          |      0.738922 |     0.771588 |  0.736489 | 0.0380422 |
| HardVotingClassifier     |      0.750898 |     0.777159 |  0.740089 | 0.0429139 |

### Confusion Matrix
*Confusion matrix* adalah tabel yang digunakan untuk mengukur kinerja dari model klasifikasi. *Confusion matrix* terdiri dari 4 bagian yaitu *true positive*, *true negative*, *false positive*, dan *false negative*.

![Confusion Matrix](/img/03_confusion%20matrix.png)

Gambar 5. Confusion Matrix
</br>
</br>
Keterangan :
- TP (True Positive) terjadi apabila wine diprediksi model sebagai wine yang baik (Positive) dan ternyata wine tersebut adalah wine yang baik (Positive).
- FN (False Negative) terjadi apabila wine diprediksi model sebagai wine yang buruk (Negative), akan tetapi ternyata wine tersebut adalah wine yang baik (Positive).
- FP (False Positive) terjadi apabila wine diprediksi model sebagai wine yang baik (Positive), akan tetapi ternyata wine tersebut adalah wine yang buruk (Negative).
- TN (True Negative) terjadi apabila wine diprediksi model sebagai wine yang buruk (Negative) dan ternyata wine tersebut adalah wine yang buruk (Negative).

Berikut adalah Confusion Matrix dari model yang telah dilatih.
```
Confusion Matrix ExtraTreesClassifier:
 [[133  36]
 [ 31 159]]

Confusion Matrix RandomForestClassifier:
 [[127  42]
 [ 33 157]]

Confusion Matrix LogisticRegression:
 [[134  35]
 [ 44 146]]

Confusion Matrix LinearSVC:
 [[132  37]
 [ 45 145]]

Confusion Matrix CalibratedClassifierCV:
 [[132  37]
 [ 45 145]]

Confusion Matrix HardVotingClassifier:
 [[133  36]
 [ 44 146]]
```

### Classification Report
*Classification report* adalah tabel yang digunakan untuk mengukur kinerja dari model klasifikasi. *Classification report* terdiri dari beberapa metrics yaitu *accuracy*, *precision*, *recall*, dan *f1-score*. Keempat metrics tersebut berasal dari *confusion matrix*.

![Classification Report](/img/04_report.webp)

Gambar 6. Rumus pada Metrics di Classification Report
</br>
</br>
Berikut adalah Classification Report dari model yang telah dilatih.
```
Classification Report ExtraTreesClassifier:
               precision    recall  f1-score   support

           0       0.81      0.79      0.80       169
           1       0.82      0.84      0.83       190

    accuracy                           0.81       359
   macro avg       0.81      0.81      0.81       359
weighted avg       0.81      0.81      0.81       359


Classification Report RandomForestClassifier:
               precision    recall  f1-score   support

           0       0.79      0.75      0.77       169
           1       0.79      0.83      0.81       190

    accuracy                           0.79       359
   macro avg       0.79      0.79      0.79       359
weighted avg       0.79      0.79      0.79       359



Classification Report LogisticRegression:
               precision    recall  f1-score   support

           0       0.75      0.79      0.77       169
           1       0.81      0.77      0.79       190

    accuracy                           0.78       359
   macro avg       0.78      0.78      0.78       359
weighted avg       0.78      0.78      0.78       359



Classification Report LinearSVC:
               precision    recall  f1-score   support

           0       0.75      0.78      0.76       169
           1       0.80      0.76      0.78       190

    accuracy                           0.77       359
   macro avg       0.77      0.77      0.77       359
weighted avg       0.77      0.77      0.77       359


Classification Report CalibratedClassifierCV:
               precision    recall  f1-score   support

           0       0.75      0.78      0.76       169
           1       0.80      0.76      0.78       190

    accuracy                           0.77       359
   macro avg       0.77      0.77      0.77       359
weighted avg       0.77      0.77      0.77       359



Classification Report HardVotingClassifier:
               precision    recall  f1-score   support

           0       0.75      0.79      0.77       169
           1       0.80      0.77      0.78       190

    accuracy                           0.78       359
   macro avg       0.78      0.78      0.78       359
weighted avg       0.78      0.78      0.78       359

```

Dalam studi kasus ini, selain melihat nilai akurasi dari *train data* dan *test data* juga melihat akurasi pada *cross validation* untuk mengetahui apakah model yang dibuat memiliki kemungkinan overfitting atau underfitting dengan melihat akurasi rata-rata dan standar deviasi. Model yang baik adalah model yang memiliki akurasi rata-rata yang tinggi dan standar deviasi yang rendah. 

Selain itu, model diharapkan mampu memberikan prediksi klasifikasi kualitas wine secara True dan akan lebih diterima apabila letak error model lebih banyak berada pada False Negative (FN) dibanding dengan False Positive (FP). Hal tersebut dikarenakan apabila kita melihat dari sisi bisnis, lebih baik model memprediksi wine yang baik sebagai wine yang buruk daripada sebaliknya. Hal tersebut dikarenakan kita dapat melakukan pengecekan kembali terhadap wine tersebut.

Kesimpulan yang diperoleh dari hasil analisis dan pemodelan *machine learning* untuk kasus ini yakni model yang digunakan untuk memprediksi kualitas wine adalah model dengan algoritma **Hard Voting Classifirer**. Model ini memiliki performa ketepatan klasifikasi yang cukup baik jika melihat pada nilai akurasi, rata-rata akurasi *cross validation*, dan *confusion matrix*. Namun meskipun demikian, masih perlu adanya peningkatan lebih lanjut untuk meminimalisir kesalahan klasifikasi.
