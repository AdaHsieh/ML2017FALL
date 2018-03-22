
The requirements of this assignment are as follows:
=============

- hw1.sh
>Python3.5+ required
Only 
- (1)numpy 
- (2)scipy 
- (3)pandas are allowed

numpy.linalg.lstsq is forbidden.
Please handcraft "linear regression" using Gradient Descent
beat public simple baseline
For those who wish to load model instead of running whole training precess:
please upload your training code named train.py
as long as there are Gradient Descent Code in train.py, it's fine


- hw1_best.sh
>Python3.5+ required
any library is allowed
meet the higher score you choose in kaggle


Data 簡介
--------------

- train.csv : 每個月前20天每個小時的氣象資料(每小時有18種測資)。共12個月。
- test.csv : 排除train.csv中剩餘的資料，取連續9小時的資料當feature，預測第10小時的PM2.5值。總共取240筆不重複的test data。
