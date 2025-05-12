# Milk-GLCM
Milk classification using GLCM

Ekstraksi fitur dilakukan untuk menghitung nilai contrast, dissimilarity, homogeneity, energy, dan correlation pada jarak (distance) 1, 2, dan 3, serta pada sudut (angle) 0°, 45°, 90°, dan 135°. Dengan demikian, total fitur yang diekstraksi untuk setiap gambar mencapai 60 fitur.

Selanjutnya dengan menggunakan Analysis of Variance mencari fitur yang paling sesuai untuk digunakan untuk klasifikasi. Dilalukan tes untuk 10 sampai 30 fitur teratas

Hasil percobaan terhadap seleksi fitur. 

| Model        | All Features | Top 10 Features | Top 20 Features | Top 30 Features |
|--------------|--------------|--------|--------|--------|
| LightGBM     | 0.6167       | 0.5833 | 0.65   | 0.68   |
| Random Forest| 0.57         | 0.70   | 0.63   | 0.65   |
| SVM          | 0.46         | 0.39   | 0.39   | 0.38   |
