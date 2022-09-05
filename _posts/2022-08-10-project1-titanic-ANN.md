---
title:  "Titanic - ANN"
excerpt: "Titanic - Machine Learning from Disaster"

categories:
  - project
tags:
  - [project]

toc: true
toc_sticky: true
toc_label : "C O N T E N T S"
 
date: 2022-07-11
last_modified_at: 2022-08-10
---  

#### 전처리 시작 + 파일 불러오기


```python
import os                        # 내 컴퓨터에 있는 파일을 불러오기 위한 라이브러리
import numpy as np               # 수학계산 함수를 불러오는 라이브러리
import pandas as pd              # 파일을 불러올 때 사용하는 라이브러리
import matplotlib.pyplot as plt  # 시각화 그래프를 위한 라이브러리 
import seaborn as sns            # 시각화 그래프를 위한 라이브러리 
import sklearn                   # 머신러닝 라이브러리
```


```python
plt.style.use('seaborn')
sns.set(font_scale=2.5)
```

matplotlib의 기본 scheme 말고 seaborn sheme을 세팅하고, 일일이 그래프의 font_size를 지정할 필요 없이 seaborn의 font_size를 사용하면 편리합니다.


```python
file_directory = 'C:/Users/komos/pytorch/project/titanic/'
os.listdir(file_directory)
```




    ['.ipynb_checkpoints',
     'gender_submission.csv',
     'Kaggle - Titanic.ipynb',
     'test.csv',
     'train.csv']



파일이 들어있는 경로를 저장하고, 경로에 있는 파일을 확인합니다.


```python
df_train = pd.read_csv(file_directory + 'train.csv')
df_test = pd.read_csv(file_directory + 'test.csv')
df_submit = pd.read_csv(file_directory + 'gender_submission.csv')
```

train, test, submit 파일들을 따로 변수에 저장합니다.  

#### 파일 확인


```python
df_train.shape, df_test.shape, df_submit.shape
```




    ((891, 12), (418, 11), (418, 2))



파일의 크기를 보시면 train은 (891, 12), test는 (418, 11), submit은 (418, 2)입니다.  
test 데이터의 열이 11인 이유는 우리가 찾고자하는 survival 칼럼이 없습니다.  
그리고 마지막 submit 데이터는 passengerid와 Survived칼럼만 있습니다.


```python
df_train.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')



train 데이터의 칼럼은 총 12개이고, 학습해야할 칼럼은 11개입니다. 하지만 여기서 질적칼럼 중 생존률과 연관성이 적은 칼럼은 제외합니다.  
참고로 여기서 칼럼은 feature(특징)으로도 불립니다.


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



`.head()` 함수를 사용하여 데이터의 초반 부분을 간단히 볼 수 있습니다. 이렇게 관측하여 각 칼럼들의 데이터가 어떤 형태인지를 파악할 수 있습니다.


```python
df_train.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



train 데이터의 각 칼럼의 데이터타입입니다. 양적 데이터는 그대로 학습해도 되지만 질적 데이터는 양적 데이터로 변환해야합니다.  
여기서는 Name, Sex, Ticket, Cabin, Embarked 칼럼이 이에 해당됩니다.  
하지만 pclass, passengerId, SibSp, Parch는 int64형테이지만 속을 들여다 보면 질적데이터와 같습니다. 그래서 이렇게 데이터 타입을 찾는 것 보다는 문제에서 데이터 형태를 소개하는 글을 보시는게 더 좋습니다.  

#### 파일 결측값 확인


```python
df_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



`.describe()`함수를 이용하여 통계량을 알 수 있습니다.  
Passengerid의 count는 891이지만 Age의 count는 714로 177개의 결측값이 있다는 것을 알 수 있습니다.


```python
df_test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



test 데이터에서는 Age와 Fare 칼럼이 각각 86, 1개의 결측값이 있다는 것을 알 수 있습니다.  
이렇게 `.describe()`를 이용하여 결측값을 확인할 수 있지만 더 간편하게 알 수 있는 방법이 있습니다.


```python
df_train.isnull().sum() / df_train.shape[0]
```




    PassengerId    0.000000
    Survived       0.000000
    Pclass         0.000000
    Name           0.000000
    Sex            0.000000
    Age            0.198653
    SibSp          0.000000
    Parch          0.000000
    Ticket         0.000000
    Fare           0.000000
    Cabin          0.771044
    Embarked       0.002245
    dtype: float64



이걸 보시면 train 데이터에는 Age에서 19%의 결측값과 Cabin에서 77% 이 있다는걸 알 수 있습니다.


```python
df_test.isnull().sum() / df_test.shape[0]
```




    PassengerId    0.000000
    Pclass         0.000000
    Name           0.000000
    Sex            0.000000
    Age            0.205742
    SibSp          0.000000
    Parch          0.000000
    Ticket         0.000000
    Fare           0.002392
    Cabin          0.782297
    Embarked       0.000000
    dtype: float64



똑같이 test 데이터에서는 Age에서 20%, Cabin에서 78%의 결측값이 있다는걸 알 수 있습니다.  
이렇게 Age와 Carbin에서 결측값이 나왔습니다. 여기서 우리는 두 가지 선택지가 있습니다. 
1. 칼럼이 target value와 연관성이 높다면 칼럼과 연관성이 높은 다른 칼럼들을 이용하여 결측값을 채워준다.
2. 칼럼이 target value와 연관성이 없다면 학습할 때 제외한다.  
또한 칼럼에 결측값이 많다면 제외할 것입니다.

#### 전처리 시작

이제 target value(Survived)의 분포의 균형성과 질적 데이터와 target value의 연관성에 대해 알아 보겠습니다.


```python
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train['Survived'].value_counts().plot.pie(explode = [0, 0], autopct = '%1.1f%%', ax = ax[0], shadow = True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data = df_train, ax = ax[1])
ax[1].set_title('Count plot - Suvived')

plt.show()
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![output_24_1](https://user-images.githubusercontent.com/60602671/183929916-c1ba9b27-195b-4252-b88a-dd299a7f4259.png)
    



1번째 줄 : f에는 전체 Figure 사이즈, ax에는 subplot의 리스트가 들어있습니다.  
`f : Figure(1296x576), ax = [<AxeSubplot:> <AxeSubplot:>]`  

2번째 줄 : pie의 매개변수 중 explode는 pie 그래프를 조각하고 원점과 떨어지는 거리를 나타냅니다. Survived 칼럼은 0, 1 두개 이므로 리스트에 두 개의 거리를 입력했습니다. autopct는 조각의 크기를 보여주게합니다.  

target value(Survived)의 분포를 보면 비교적 균형적이라는 것을 알 수 있습니다. 만약 데이터가 불균형적이라면 데이터를 또 다르게 처리해야합니다. 이에 관한 것은 다음 포스팅에 소개하겠습니다. [참고글](https://casa-de-feel.tistory.com/15), [참고글](https://3months.tistory.com/414)  

#### 질적 데이터와 target value 연관성 
1. Pclass - Survived



```python
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>136</td>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>87</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>372</td>
      <td>119</td>
      <td>491</td>
    </tr>
    <tr>
      <th>All</th>
      <td>549</td>
      <td>342</td>
      <td>891</td>
    </tr>
  </tbody>
</table>
</div>



`pd.cosstab()`을 사용하면 질적 데이터 칼럼에 대하여 교차분석을 하여 행, 열 요인들을 기준별로 빈도를 세어서 도수분포표를 만들어줍니다. 여기서 margins 매개변수를 `True`로 해주면 빈도수의 총합을 추가로 나타내줍니다.  

이 도수분포표를 보면 1등급에서 사망한 사람은 적고, 2등급은 비슷하며 3등급에서는 사망한 사람이 많다는 것을 알 수 있습니다. 즉 Pclass와 Survived는 연관성이 있다는 것을 알 수 있습니다.


```python
df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
```




    <AxesSubplot:xlabel='Pclass'>




    
![output_28_1](https://user-images.githubusercontent.com/60602671/183930107-ba560549-8634-4183-8982-478e50778172.png)
    


`.groupby()`는 기준이 된다고 생각하시면 됩니다. 즉 Pclass를 기준으로 Survived의 평균을 보여준다는 느낌입니다.   
이로써 Pclass 칼럼은 target value와 연관성이 높다는 것을 알았습니다.  

2. Sex - Survived


```python
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Sex - Survived')
sns.countplot('Sex', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Sex : Survived vs Dead')
plt.show()
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![output_30_1](https://user-images.githubusercontent.com/60602671/183930201-ac6eba87-a4a9-4674-977b-e4b1e77979fd.png)
    


우리는 남성과 여성의 생존비율과 생존 수를 알아보았습니다. 확실히 여성이 남자보다 많이 생존했다는 것을 알 수 있어서 Sex 칼럼 또한 target value와 연관성이 있다는 것을 알 수 있습니다.  

그럼 과연 Sex와 Pclass에 관하여 Survived는 어떻게 달라지는지 알아봅시다.


```python
sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = df_train, size = 6, aspect = 1.5)
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\categorical.py:3723: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <seaborn.axisgrid.FacetGrid at 0x1aa147030d0>




    
![output_32_2](https://user-images.githubusercontent.com/60602671/183930308-0d4e2484-cbbb-41f3-89fd-92255af2baa1.png)

    


`sns.factorplot()`은 다양한 그래프를 그릴 수 있는 함수입니다. `kind` 매개변수를 이용하여 다양한 그래프를 그릴 수 있으며 기본값은 점 그래프입니다. `hue` 매개변수는 지정된 칼럼의 데이터를 색을 입힙니다. `aspect` 매개변수는 가로 세로의 비율을 나타냅니다. 주의할 점은 실수형태의 입력값을 받습니다. 더욱 다양한 매개변수들은 [seaborn factorplot](https://www.geeksforgeeks.org/python-seaborn-factorplot-method/)에서 확인 가능합니다.

이 그래프를 보면 남성과 여성 모두 등급의 차이에 따라 생존률이 확연히 차이가 나는 것을 알 수 있습니다.  

3. Family - Survived, Family = SibSp + Parch  
SibSp와 Parch는 모두 가족관계에 관련된 칼럼입니다. 그래서 이 두 칼럼을 합쳐 Family 칼럼을 새로 만들어봅시다.  
주의할 점은 칼럼을 바꿀 때 df_train 뿐만아니라 df_test 또한 바꿔야합니다.


```python
df_train['Family'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1

sns.countplot(df_train['Family'])
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='Family', ylabel='count'>




    
![output_34_2](https://user-images.githubusercontent.com/60602671/183930391-93a24f46-5717-4c52-bea2-ddcc595389d0.png)
    


여기서 `Family`칼럼을 만들 때 + 1을 한 이유는 가족이 아무도 없을 때는 혼자이니까 + 1 을 했습니다.  
이렇게 Family 칼럼을 만들었습니다. 이제 Family 칼럼과 Survived 칼럼을 비교해보자.  

위 막대 그래프를 살펴보면 혼자 온 사람이 제일 많다는걸 알 수 있습니다.


```python
sns.barplot('Family', 'Survived', data = df_train, ci = None)
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='Family', ylabel='Survived'>




    
![output_36_2](https://user-images.githubusercontent.com/60602671/183930457-43daa5bd-33c6-4148-8b59-a6c5aaa4f855.png)

    


`ci`매개변수는 3가지 옵션(숫자, 'sd', None)을 가질 수 있습니다. 숫자는 신뢰구간(%), 'sd'는 표준편차, None은 표현하지 않음을 의미합니다.  

위 막대 그래프들을 살펴보면 4명일 때 생존률이 제일 높고, 4명을 기준으로 많아지거나 적어지면 생존률이 떨어진다는 것을 알 수 있습니다.  
그래서 우리는 1 ~ 3, 4, 5 ~ 이렇게 3분류로 나눠보겠습니다.


```python
for i in range(1, 12):
    if 1 <= i <= 3:
        df_train['Family'].replace({i : 1}, inplace = True)
        df_test['Family'].replace({i : 1}, inplace = True)
    elif i == 4:
        df_train['Family'].replace({i : 2}, inplace = True)
        df_test['Family'].replace({i : 2}, inplace = True)
    elif i > 4:
        df_train['Family'].replace({i : 3}, inplace = True)
        df_test['Family'].replace({i : 3}, inplace = True)
    
```

4. Age - Survived  
Age 칼럼에는 결측값이 있었습니다. 이를 채우기 위해 먼저 Age와 연관성이 높은 다른 칼럼들을 찾은 후, 비슷한 데이터가 있으면 그 데이터의 나이를 결측값에 대신합니다. 만약 없다면 중간값으로 대체합니다. 이를 수행하기 위해서는 우선 질적 데이터를 양적 데이터로 바꿔야합니다. (Sex, Embarked) 그런데 Embarked 칼럼에도 결측값이 있습니다. 이는 수량이 비교적 매우 적기 때문에 Southampton으로 변환하겠습니다.


```python
df_train['Embarked'].fillna('S', inplace = True)
df_test['Embarked'].fillna('S', inplace = True)

df_train['Embarked'].replace({'C' : 0, 'Q' : 1, 'S' : 2}, inplace = True)
df_test['Embarked'].replace({'C' : 0, 'Q' : 1, 'S' : 2}, inplace = True)

df_train['Sex'].replace({'female' : 1, 'male' : 0}, inplace = True)
df_test['Sex'].replace({'female' : 1, 'male' : 0}, inplace = True)
```

`fillna()`함수를 이용하여 Embarked 칼럼에서 결측값을 모두 S로 바꿨습니다. 여기서 `inplace` 매개변수를 `True`로 할 경우 기존의 데이터프레임이 수정됩니다. 그 후 Embarked는 C : 0, Q : 1, S : 2 로 변경되었고, Sex는 female : 1, male : 0으로 변경됬습니다.  
이제 Age와 연관성이 있는 칼럼을 찾아봅시다.


```python
heatmap_data = df_train[['Age', 'Pclass', 'Sex', 'Fare', 'Embarked', 'Family']]
plt.figure(figsize=(15,15))
sns.heatmap(heatmap_data.corr(), annot=True, square=True)
```




    <AxesSubplot:>




    
![output_42_1](https://user-images.githubusercontent.com/60602671/183930545-9beb3e64-585e-43ca-a8fa-3eccdf499c02.png)
    


`corr()`함수는 칼럼 사이의 상관관계를 보여줍니다. 그리고 heatmap의 `annot` 매개변수는 각 셀에 값을 표기하고, `square` 매개변수는 heatmap을 정사각형 형태로 만들어줍니다.  

heatmap.corr()은 0일때 가장 연관성이 낮고, 절댓값이 1에 가까울수록 연관성이 높다고 판단합니다. 그래서 Pclass와 Family 칼럼이 Age와 상대적으로 연관성이 높다고 판단가능합니다. 그러므로 연관성이 높은 Pclass와 Family로 Age의 결측값을 채워봅시다.



```python
pclass_value = [1, 2, 3]
Family_value = [1, 2, 3, 4, 5, 6, 7, 8, 11]

for i in pclass_value:
    for j in Family_value:
        age_guess_train = df_train.loc[(df_train['Pclass'] == i) & (df_train['Family'] == j), 'Age'].dropna().median()
        df_train.loc[(df_train['Pclass'] == i) & (df_train['Family'] == j) & (df_train['Age'].isnull()), 'Age'] = age_guess_train
        
        age_guess_test = df_test.loc[(df_test['Pclass'] == i) & (df_test['Family'] == j), 'Age'].dropna().median()
        df_test.loc[(df_test['Pclass'] == i) & (df_test['Family'] == j) & (df_test['Age'].isnull()), 'Age'] = age_guess_test
```

`.loc()`함수는 인덱스를 기준으로 행 데이터를 가져옵니다. 즉 Pclass와 Family의 데이터가 있고, Age에서 결측값이 없는 데이터를 가져옵니다. 여기서 그 Age 값의 중간값으로 가져옵니다. 그런다음 Pclass와 Family의 데이터가 있고, Age가 결측값인 데이터에 그 중간값을 대입하는 것입니다.  

5. Fare - Survived
Fare과 Survived의 연관성도 한번 보겠습니다. Fare은 배 표값인데 표값이 비싸면 분명 상류층 사람이 될 테고, 그럼 생존률이 높지 않을까라는 예측을 할 수 있습니다. df_test 데이터에는 Fare 칼럼에 결측값이 있어 만약 df_train에서 Fare과 Survived가 연관성이 있다면 결측값을 중간값으로 대체하고, 없다면 drop하겠습니다.


```python
for i in range(len(df_train['Fare'])):
    df_train['Fare'][i] = int(df_train['Fare'][i])
```

    C:\Users\komos\AppData\Local\Temp\ipykernel_13192\267103575.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_train['Fare'][i] = int(df_train['Fare'][i])
    


```python
sns.factorplot('Fare', 'Survived', data = df_train, size = 6, aspect = 1.5)
```

    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\categorical.py:3723: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    C:\Users\komos\anaconda3\envs\LSJ\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <seaborn.axisgrid.FacetGrid at 0x1aa14dbc550>




    
![output_47_2](https://user-images.githubusercontent.com/60602671/183930619-bd597aa6-d60e-4ed1-987e-1dfeccbbdc28.png)

    


Fare의 데이터는 다양하게 이루어져 있어 정수로 바꿔주고 Survived와 관계를 살펴봤습니다. 근데 다양한 데이터에 생존률도 다양해서 Fare칼럼은 연관성이 없다고 생각했습니다. 그리하여 Fare 칼럼은 drop 하겠습니다.

그럼 drop할 칼럼은 PassengerId, Name, SibSp, Parch, Carbin, Fare, Ticket 입니다. (Carbin은 결측값이 너무 많아 제외하겠습니다.)


```python
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



`drop`함수는 pandas에 DataFrame.drop()으로 사용하실 수 있습니다. drop의 매개변수들 중 `axis`는 0과 1을 입력받는데, 0은 index를 drop한다고 하고, 1은 columns를 drop한다고 합니다. 코드에서는 1이므로 columns를 drop합니다. `inplace`는 False와 True가 있는데 False를 하면 복사를 합니다. 즉 원래의 데이터에 영향을 끼치지 않습니다. 하지만 True를 한다면 원래 데이터를 변환합니다.

# 모델링

ANN의 모델링을 시작하겠습니다. pytorch로 작성할 것이고, input dim는 데이터 칼럼 개수인 5개로, hidden layer는 2개(각각 노드 64, 32개)로 하고, output dim는 Survived를 예측하므로 노드 1개로 이루어지게 하겠습니다.

GPU연산을 위해 cuda와 cuDNN, tensorflow-gpu, pytorch를 다운했습니다. [poeun 블로그](https://ingu627.github.io/tips/install_cuda2/)에서 참고했습니다.


```python
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
```


```python
y = df_train['Survived']
df_train.drop(['Survived'], axis = 1, inplace = True)
x = df_train
```

먼저 데이터를 x(feature columns)와 y(target columns)로 구분하겠습니다.


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

    (712, 5) (712,) (179, 5) (179,)
    

sklearn.model_selection의 `train_test_split`은 배열과 행렬을 랜덤으로 train, test로 나눕니다. 매개변수로는 일단 배열또는 행렬을 입력받고, `test_size`는 0.0에서 1.0 사이의 실수형태를 입력받으며 test set의 분포를 나타냅니다. 즉 test_size가 0.20이므로 전체의 20%를 test set으로 사용한다는 의미입니다. `stratify`는 데이터를 계층된 방식으로 분리한다는 의미입니다. 즉 stratify가 y이므로 Survived를 기점으로 분리하겠다는 의미입니다. 마지막으로 `random_state`는 분할하기전 shuffle을 적용합니다. 여기서 2의 값을 줬기 때문에 2번의 shuffle을 적용합니다.


```python
x_train = torch.Tensor(x_train.values)
y_train = torch.Tensor(y_train.values)

x_test = torch.Tensor(x_test.values)
y_test = torch.Tensor(y_test.values)
```

`torch.Tensor`는 데이터의 타입을 torch의 Tensor형태로 변환시켜줍니다. 


```python
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)
```


```python
train_load = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_load = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)
```

torch.utils.data의 `TensorDataset`과 `TensorDataLoader`를 사용하면 방대한 양의 데이터를 미니배치 단위로 처리할 수 있고, 데이터를 무작위로 섞음으로써 효율성을 향상시킬 수 있습니다. TensorDataset은 전체적으로 학습 데이터, 테스트 데이터를 구분해주고, TensorLoader는 `batch_size`를 통해 배치사이즈를 결정하고, `shuffle`을 통해 무작위로 섞어줍니다.


```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        logits = self.linear_relu_stack(x)
        return logits
```

코드를 하나하나씩 파헤쳐가며 함수들을 설명하겠습니다. 우선 `nn.Sequential`은 여러개의 함수를 묶어 놓기 좋은 함수이며, 데이터가 하나씩 입력받았을 때 사용하기 좋습니다. 그 다음 `nn.Linear`함수는 선형회귀모형을 구현해줍니다. input_dim이 5이고 output이 64이므로 5개를 입력받아 64개로 나온다는 의미입니다. 그 뒤 `nn.ReLU` 함수는 선형함수를 비선형함수로 변환시키는 목적이있습니다. 마지막으로 `nn.Sigmoid`함수는 입력값을 0과 1사이의 값으로 변환시킵니다. 뒤에 나오겠지만 우리는 이 값을 소수점 첫째자리에서 반올림을 하여 0과 1을 얻을 것입니다.  

0 index의 데이터를 위 함수 순서대로 실행해봤습니다. 결과는 맨 아래에서 확인 가능합니다.


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

device_gpu = torch.device('cuda')
```

    Using cuda device
    

GPU연산을 위해 device에 cuda를 설정했습니다.


```python
input_dim = 5
output_dim = 1

model = NeuralNetwork(input_dim, output_dim).to(device)
print(model)
```

    NeuralNetwork(
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=5, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=32, bias=True)
        (3): ReLU()
        (4): Linear(in_features=32, out_features=1, bias=True)
        (5): Sigmoid()
      )
    )
    

# 손실함수 + 매개변수 최적화


```python
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
```

loss function으로는 `MSELoss`를 사용하겠습니다. MSELoss의 수식은 제 이전 포스팅 [ANN 논문 리뷰 2](https://komos0509.github.io/review/review2/)에서 확인하실 수 있습니다. 

또한 최적화 함수로는 torch.optim의 `SGD(확률적 경사하강법)`함수를 이용했습니다. 경사하강법의 수식은 제 이전 포스팅 [ANN 논문 리뷰 1](https://komos0509.github.io/review/review1/)에서 확인하실 수 있습니다. SGD의 매개변수로는 최적화를 시킬 대상인 model.parameters()가 있고, `lr`은 학습률입니다. 제 이전 포스팅에서는 $\mu$에 해당합니다.

# 성능확인


```python
from tqdm import tqdm
from sklearn.metrics import accuracy_score
```

이렇게 오랜 시간동안 작동하는 코드가 있을 때 얼마만큼 진행되었는지를 모릅니다. 그것을 방지하여 `tqdm`라이브러리를 이용하면 매우 간단하고 효과적으로 진행상황을 볼 수 있습니다. 사용법은 for 문의 in 구문을 tqdm으로 감싸면 됩니다.


```python
epoch = 10

for i in range(1, epoch + 1):
    train_loss = []
    test_loss = []
    accuracy = []
    model.train()
    for idx, (X_batch, Y_batch) in enumerate(tqdm(train_load)):
        X_batch = X_batch.to(device_gpu)
        Y_batch = Y_batch.to(device_gpu)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred.squeeze(), Y_batch)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    model.eval()
    for idx, (X_batch, Y_batch) in enumerate(tqdm(test_load)):
        X_batch = X_batch.to(device_gpu)
        Y_batch = Y_batch.to(device_gpu)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred.squeeze(), Y_batch)
        test_loss.append(loss.item())
        
        y_pred = np.round(y_pred.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        acc = accuracy_score(Y_batch.detach().cpu().numpy(), y_pred)
        accuracy.append(acc)
        
    print('epoch {}, loss {}, Accuracy {}'.format(i, np.mean(test_loss), np.mean(accuracy)))
```

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 346.49it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 230.88it/s]
    

    epoch 1, loss 0.19587704042593637, Accuracy 0.6884191176470589
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 343.78it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 273.43it/s]
    

    epoch 2, loss 0.19602894286314645, Accuracy 0.6961805555555555
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 348.34it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 272.74it/s]
    

    epoch 3, loss 0.2124378482500712, Accuracy 0.6922998366013072
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 334.23it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.68it/s]
    

    epoch 4, loss 0.20742933948834738, Accuracy 0.687091503267974
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 316.64it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.66it/s]
    

    epoch 5, loss 0.16620522240797678, Accuracy 0.7653186274509803
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 334.22it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.66it/s]
    

    epoch 6, loss 0.20000251630942026, Accuracy 0.7142565359477123
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 388.15it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.67it/s]
    

    epoch 7, loss 0.2212099681297938, Accuracy 0.6282679738562091
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 364.62it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.67it/s]
    

    epoch 8, loss 0.20218568543593088, Accuracy 0.7249795751633986
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 353.89it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 273.48it/s]
    

    epoch 9, loss 0.3092605272928874, Accuracy 0.6479779411764706
    

    100%|█████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 376.01it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 250.65it/s]

    epoch 10, loss 0.17959009110927582, Accuracy 0.7431576797385621
    

    
    

우선 `model.train()`이라는 코드를 살펴봅시다. nn.Module에는 train time과 evaluate time에 수행하는 작업을 스위칭할 수 있도록 함수를 구현해 놓았습니다. 즉 model.eval()을 하면 evaluate과정에서 사용하지 않을 layer는 작동을 멈춥니다. 그리고 다시 model.train을 하여 전체 layer를 가동시킵니다.  

`enumerate`함수를 보면 for문의 in구문 뒤의 목록에서 index정보와 원소 정보를 같이 얻기 위해 사용하는 함수입니다. enumerate함수를 in 구문에 감싸주면 (인덱스, 원소) 이렇게 tuple 형태로 나타납니다. 코드에서는 idx는 인덱스, X_batch는 feature data, Y_batch는 target data입니다. 

`X_batch = X_batch.to(device_gpu)`는 train_load나 test_load에서 X_batch나 Y_batch는 CPU연산을 하고 있습니다. 하지만 model을 적용하기 위해서는 이를 GPU연산을 시켜야 하므로 코드를 작성했습니다..

`optimizer.zero_grad(), loss.backward(), optimizer.step()` 이는 매개변수들을 최적화 시키기 위해 작성했습니다. 먼저 optimizer.zero_grad()는 매개변수의 최적화를 위해 backward()를 할 때 누적됨을 방지하기위해 초기화를 시키는 것입니다. 그리고 backward()를 이용해 최적화를 시킨 뒤, optimizer.step()을 이용해 이를 적용시킵니다.

`loss = loss_fn(y_pred.squeeze(), Y_batch)`는 loss를 구하기 위해서 작성되었습니다. 여기서 y_pred의 크기는 (64, 1), (64, 1), (51, 1)입니다. 이는 먼저 train_load나 test_load의 batch size를 64로 했기 때문에 이렇게 나뉘어 졌습니다. 여기서 우리는 y_pred의 값과 실제 값 Y_batch와의 비교를 위해 뒷 차원인 1을 제거해야합니다. 이를 위하여 `.squeeze()`를 사용하여 크기가 1인 차원을 제거합니다.  
그 뒤 `loss.item()`을 이용하여 데이터 값만 test_loss에 저장합니다.

`y_pred = np.round(y_pred.detach().cpu().numpy())`는 예측했던 y_pred를 Survived의 데이터인 0과 1로 맞추기 위하여 np.round를 이용했습니다. 여기서 detach()를 사용하면 Tensor를 복사하는데 gradient 전파가 안되는 것이 특징입니다. 그리고 numpy 연산을 하기 위해서는 GPU연산이 아닌 CPU연산을 적용해야 함으로 .cpu()를 통해 CPU연산을 적용시켰습니다.

`accuracy_score`는 일반적으로 정확도를 평가해줍니다. 정답 배열과 예측값 배열을 넣으면 정확도를 출력해줍니다.

# 마무리
이제 마무리로 만들었던 모델을 df_test에 적용시켜 제출할 코드까지 파일화하여 제출하겠습니다.  


```python
import torch
import torchvision.models as models
```


```python
x_df_test = torch.Tensor(df_test.values)
for idx, X_batch in enumerate(tqdm(x_df_test)):
    X_batch = X_batch.to(device_gpu)
    y_pred = model(X_batch)
    y_pred = np.round(y_pred.detach().cpu().numpy())
    df_submit['Survived'][idx] = y_pred
```

    100%|███████████████████████████████████████████████████████████████████████████████| 418/418 [00:00<00:00, 867.49it/s]
    

완성된 모델을 우리가 원하는 df_test에 적용시켰습니다. 그리고 마지막으로 제출해야하는 데이터인 df_submit의 Survived 칼럼에 예측한 결과를 적용했습니다.


```python
df_submit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_submit.to_csv('simple_nn_submission.csv', index=False)
```

완성된 df_submit 데이터를 `to_csv` 함수를 이용하여 csv 파일로 저장했습니다. 여기서 `index` 매개변수를 True로 한다면 각 행의 인덱스를 표시합니다.  

아래는 제출한 결과입니다.

여담으로 61% 보다 높게 만들려면 feature의 갯수를 늘리거나, model를 더욱 복잡하게 하고, 손실함수를 변경하는 등 많은 방법이 있는 것 같습니다. 이는 추후 코드를 추가로 작성하여 비교해보겠습니다.

![result_titanic_ANN](https://user-images.githubusercontent.com/60602671/183922211-418ae841-a63f-4252-b077-f25d50c37c81.PNG)



```python
l1 = [ 3.0000,  0.0000, 20.0000,  2.0000,  1.0000]
lis = torch.Tensor(l1)
p1 = nn.Linear(5, 64)
r1 = p1(lis)
print(r1)
p2 = nn.ReLU()
r2 = p2(r1)
print(r2)
p3 = nn.Linear(64, 32)
r3 = p3(r2)
print(r3)
p4 = nn.ReLU()
r4 = p4(r3)
print(r4)
p5 = nn.Linear(32, 1)
r5 = p5(r4)
print(r5)
p6 = nn.Sigmoid()
r6 = p6(r5)
print(r6)
```

    tensor([-5.7244,  0.2506,  5.3217,  5.3299,  3.2708, -0.2386,  2.3790,  3.0727,
             1.2007, -0.2367,  5.7793, -5.5076,  7.0623, -1.6534, -0.8943, -0.4076,
            -3.2156, -6.9080,  6.4254, -2.5314, -4.1812,  1.2426,  0.0871,  7.9831,
            -0.5065,  4.4858,  2.0065, -9.5977,  7.5920, -7.5404,  7.2085, -8.1296,
            -6.3328,  3.7332,  4.6758, -2.5522,  5.8978,  3.3226, -0.1276,  7.7554,
            -5.0113, -7.2521,  1.5864, -1.0281, -7.7842,  6.5348,  5.1830, -9.0264,
             3.7179,  6.5366, -0.7453, -9.6336, -6.9403,  6.0495,  0.7419,  0.5408,
             4.5726,  7.7100, -0.7137, -3.1799,  7.4947,  7.2300,  5.9774,  5.8581],
           grad_fn=<AddBackward0>)
    tensor([0.0000, 0.2506, 5.3217, 5.3299, 3.2708, 0.0000, 2.3790, 3.0727, 1.2007,
            0.0000, 5.7793, 0.0000, 7.0623, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            6.4254, 0.0000, 0.0000, 1.2426, 0.0871, 7.9831, 0.0000, 4.4858, 2.0065,
            0.0000, 7.5920, 0.0000, 7.2085, 0.0000, 0.0000, 3.7332, 4.6758, 0.0000,
            5.8978, 3.3226, 0.0000, 7.7554, 0.0000, 0.0000, 1.5864, 0.0000, 0.0000,
            6.5348, 5.1830, 0.0000, 3.7179, 6.5366, 0.0000, 0.0000, 0.0000, 6.0495,
            0.7419, 0.5408, 4.5726, 7.7100, 0.0000, 0.0000, 7.4947, 7.2300, 5.9774,
            5.8581], grad_fn=<ReluBackward0>)
    tensor([-5.3503e-01,  1.9197e+00,  3.8550e+00,  2.6925e+00, -2.4141e+00,
             8.9154e-01, -3.4031e+00, -3.9079e+00, -2.6029e+00, -3.4837e-02,
             2.4478e-04, -9.8228e-01,  1.2042e-01, -5.6451e+00,  2.0114e+00,
            -1.8729e+00,  1.2635e-01, -1.4052e+00, -2.6635e+00, -1.5465e-01,
            -1.6492e+00, -3.3557e+00,  5.1659e-01,  2.8576e+00, -1.6086e+00,
            -6.1692e-01,  1.6246e+00,  6.9056e-01,  1.0274e+00, -1.3613e+00,
             2.9564e+00,  1.0681e+00], grad_fn=<AddBackward0>)
    tensor([0.0000e+00, 1.9197e+00, 3.8550e+00, 2.6925e+00, 0.0000e+00, 8.9154e-01,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4478e-04, 0.0000e+00,
            1.2042e-01, 0.0000e+00, 2.0114e+00, 0.0000e+00, 1.2635e-01, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.1659e-01, 2.8576e+00,
            0.0000e+00, 0.0000e+00, 1.6246e+00, 6.9056e-01, 1.0274e+00, 0.0000e+00,
            2.9564e+00, 1.0681e+00], grad_fn=<ReluBackward0>)
    tensor([0.5412], grad_fn=<AddBackward0>)
    tensor([0.6321], grad_fn=<SigmoidBackward0>)
    

0 index에서 모델(NeuralNetwork)의 함수들을 실행했을 때의 결과입니다. Linear에서 무작위이기 때문에 결과값이 달라질 수 있습니다.
