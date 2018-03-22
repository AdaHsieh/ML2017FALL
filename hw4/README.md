
Task - Text Sentiment Classification
===============

Requirements
===============
>In this assignment, you will practice to build RNN model/BOW model.


> 
- 1.Build Convolution Neural Network
- 2.Build Deep Neural Network
- 3.Analyze the Model by Confusion Matrix
[Analysis] Plot the prediction into confusion matrix and describe what you observed.
- 4.Analyze the Model by Plotting the Saliency Map
[Analysis] Plot the saliency map of original image to see which part is important when classifying</
- 5.Analyze the Model by Visualizing Filters

Data
----------------
- 為twitter上收集到的推文，每則推文都會被標注為正面或負面
- 1 ：正面, 0：負面

Preprocessing the sentences
----------------
> 
- 先建立字典，字典內含有每一個字所對應到的index
- 利用Word Embedding來代表每一個單字
- 並藉由RNN model 得到一個代表該句的vector
- 可直接用bag of words(BOW)的方式獲得代表該句的vector

> Word Embedding
- 用一個向量(vector)表示字(詞)的意思

> 1-of-N encoding
> 
Bag of Words (BOW)
- BOW的概念就是將句子裡的文字變成一個袋子裝著這些詞的方式表現，
這種表現方式不考慮文法以及詞的順序。

Semi-Supervised learning
-----------------
- semi-supervised 簡單來說就是讓機器自己從unlabel data中找出label，
而方法有很多種，這邊簡單介紹其中一種比較好實作的方法 Self-Training。

- Self-Training：
把train好的model對unlabel data做預測，並將這些預測後的值轉成該筆
unlabel data的label，並加入這些新的data做training。
你可以調整不同的threshold、或是多次取樣來得到比較有信心的data。
ex：設定pos_threshold=0.8，只有在prediction>0.8的data才會被標上1的label
		

> [(link)](https://ntumlta.github.io/ML-Assignment4/index.html)