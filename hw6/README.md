
Task -  Unsupervised learning & dimension reduction 
===============
[(link)](https://ntumlta.github.io/2017fall-ml-hw6/)



Task Introduction
-----------------
> Part A: PCA of colored faces

> Part B: Visualization of Chinese word embedding

> Part C: Image clustering



PCA of colored faces

>學習用 numpy 實做 PCA 以達到 dimensionality reduction 的目的
跟以往不同，這次是對彩色的臉做 PCA。

>數據集來自 Aberdeen University 的 Prof. Ian Craw，並經過助教們的挑選及對齊，總共有 415 張 600 X 600 X 3 的彩圖。
[(data link)](https://drive.google.com/open?id=1_zD31Iglz6eTh55ushu-5dtciatuVyPy)

PCA of colored faces - reminder
>請記得先減去平均再計算 Eigenfaces, Eigenvalues

>Eigenfaces 是奇怪的顏色是正常的

>因為 Eigenfaces 會有負值，因此在畫圖時，請用以下方式轉換：

>M -= np.min(M)

>M /= np.max(M)

>M = (M * 255).astype(np.uint8)


Visualization of Chinese word embedding 
>用任何 word2vec 的套件在中文的句子上訓練中文字的 word embedding
有 python-package 的常用 word2vec 套件：gensim, word2vec, glove-python, glove

>用任何dimension reduction的演算法在二維平面上視覺化 word embedding
可以使用任何 dimension reduction 的演算法，但建議使用 TSNE

>從視覺化的結果觀察 word embedding 訓練的成果

>[(data link)](https://drive.google.com/open?id=1E5lElPutaWqKYPhSYLmVfw6olHjKDgdK)

>建議用 jieba 分詞

>因為 jieba 預設主要是簡體字，建議使用繁體分詞更好的 dict.txt.big

> - 從連結下載詞典，然後用 jieba.set_dictionary (path_to_downloaded_dict.txt.big)

>Visualization 的時候，只針對出現次數 ≥K 的詞，建議 6000 ≥ K ≥ 3000

>用 adjustText 避免圖表文字的重疊

>用 matplotlib 作圖的話，要注意中文字體的設定，否則會出現亂碼

> - keras.layers.Embedding : the user matrix and item matrix can be viewed as two embedding matrix

> - keras.layers.Flatten: the output tensor shape of embedding layer would be [batch_size,1,embedding_dim], you need this function to reshape the tensor to [batch_size,embedding_dim]

> - keras.layers.Dot : if applied to two tensors a and b of shape (batch_size, n), the output will be a tensor of shape (batch_size, 1) where each entry i will be the dot product between a[i] and b[i].

> - keras.layers.Add : add all tensors

> - keras.layers.Concatenate : concatenate two tensor



Image clustering
>目標：分辨給定的兩張 images 是否來自同一個 dataset

> - 所有的 image 都來自兩個不同的 dataset
> - 除了 image 本身之外，沒有任何 label
> - 不使用額外的 dataset (包括用額外資料 train 的 model)

>evaluation

> - F1-Score




Data format
----------------
> 總共有 140000 張 image，都是黑白圖片

> image.npy.zip

> - 輸入指令 unzip image.npy.zip，會得到一個檔案叫做 image.npy
> - 使用 np.load() 讀取 image.npy，會得到一個 140000x784 的 ndarray
> - 每一個 row 都代表一張 28x28 image


> test_case.csv

> - 每一行都有 ID, image1_index, image2_index，總共有 1,980,000 筆測資
> - ID: test case index
> - image1_index: 對應到 image.npy 裡的 row index
> - image2_index: 對應到 image.npy 裡的 row index

>sample_submission.csv

> - 第一行是 “ID,Ans”
> - 之後每一行都會有 test case ID，以及對這個 test case 的 prediction
> - 如果 test case 的兩張 image 預測後是來自同一 dataset，Ans 的地方就是 1，反之是 0




Rules
------------------
Basic : No extra dataset. No plagiarism.

Allowed toolkits for "PCA of colored faces":

- NumPy 1.13+

- scikit-image 0.13+

Allowed toolkits for "Visualization of Chinese word embedding":

you can use anything

Allowed toolkits for "Image clustering":

- python 3.5+

- tensorflow 1.3

- keras 2.0.8

- pytorch 0.2.0

- scikit-image 0.13+

- pillow 4.3.0

- scikit-learn 0.19+

- pandas 0.20+

- numpy

- scipy

- matplotlib

- h5py

- Python Standard Lib
