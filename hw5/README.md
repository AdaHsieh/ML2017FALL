
Task -  Movie Recommendation
===============

Task Introduction
-----------------
> Given the user’s rating history on items, we want to predict the rating of unseen (user,item) pairs.
> We want you to implement matrix factorization to predict the missing value on user-item matrix.


> Matrix Factorization

> Useful function in Keras

> - keras.layers.Embedding : the user matrix and item matrix can be viewed as two embedding matrix

> - keras.layers.Flatten: the output tensor shape of embedding layer would be [batch_size,1,embedding_dim], you need this function to reshape the tensor to [batch_size,embedding_dim]

> - keras.layers.Dot : if applied to two tensors a and b of shape (batch_size, n), the output will be a tensor of shape (batch_size, 1) where each entry i will be the dot product between a[i] and b[i].

> - keras.layers.Add : add all tensors

> - keras.layers.Concatenate : concatenate two tensor



> 
- 1.Build Convolution Neural Network
- 2.Build Deep Neural Network
- 3.Analyze the Model by Confusion Matrix
[Analysis] Plot the prediction into confusion matrix and describe what you observed.
- 4.Analyze the Model by Plotting the Saliency Map
[Analysis] Plot the saliency map of original image to see which part is important when classifying</
- 5.Analyze the Model by Visualizing Filters

Data format
----------------
Training data: 899873

Testing data: 100336, half private set.


- train.csv
- TrainDataID, UserID,MovieID,Rating

- test.csv
- TestDataID,UserID,MovieID

- movies.csv
- movieID::Title::Genres

- users.csv
- UserID::Gender::Age::Occupation::Zip-code

Evaluation
------------------

- RMSE



Hint
----------------

Normalize

 
- 因為Rating是介於[1, 5]之間。可以嘗試將Rating normalize到[0, 1]；或是減掉平均，除以標準差等等，方法不限。

DNN

- DNN input:舉例來說可以把user的embedding以及movie的embedding連接在一起，作為DNN的input
- DNN output: 可以把這個問題視為regression問題，又或者將1-5每種分數都視為不同類別，再去做5個類別的分類問題



> [(link)](https://ntumlta.github.io/2017fall-ml-hw5/)