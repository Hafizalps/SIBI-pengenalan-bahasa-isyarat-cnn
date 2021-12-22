# Pengenalan Bahasa Isyarat SIBI
Program pengenalan bahasa isyarat (SIBI).
Pada project ini saya mengimplementasikan CNN (Convolution Neural Network) menggunakan Keras.

### Tools yang digunakan
1. Python 3
2. OpenCV 3
3. Tensorflow
4. Keras

### Langkah-langkah menjalankan project
1. Install Python 3, Opencv 3, Tensorflow, Keras.
2. Pertama, latih model terlebih dahulu.
    ```
    python cnn_model.py
    ```
2. Sekarang untuk menguji model, Hanya perlu menjalankan recognise.py . Untuk melakukannya cukup buka terminal dan jalankan perintah berikut..
    ```
    python recognise.py
    ```
    Sesuaikan nilai hsv dari trackbar untuk mengelompokkan warna tangan Anda.
    ```
3. Untuk melakukan pengenalan bahasa isyarat secara real-time, jalankan perintah berikut..
    ```
    python recognise.py
    ```

4. Untuk membuat kumpulan data set Anda sendiri, jalankan perintah berikut..
    ```
    python capture.py
    ```





