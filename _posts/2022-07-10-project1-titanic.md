---
title:  "Titanic"
excerpt: "Titanic - Machine Learning from Disaster"

categories:
  - project
tags:
  - [project]

toc: true
toc_sticky: true
toc_label : "C O N T E N T S"
 
date: 2022-07-11
last_modified_at: 2022-07-11
---  

본 프로젝트는 Kaggle사이트에서 가져왔습니다. [사이트 들어가기](https://www.kaggle.com/competitions/titanic)  

## 프로젝트 소개  

1912년 4월 15일 타이타닉호는 빙산과 충돌하여 2224명의 승객 중 1502명의 승객이 사망하는 사고가 발생했습니다. 우리는 여기서 승객들의 정보(이름, 나이, 성별, 좌석 등등)를 이용하여 승객이 생존했는지 사망했는지를 예측하는 모델을 만들 것입니다.  

일단 데이터를 보면 test.csv, train.csv 데이터가 있습니다. train 데이터에는 승객들의 생존여부를 나타내는 Survived 칼럼이 있습니다. (test 데이터에는 없습니다.) 그래서 train 데이터를 이용하여 모델을 구축하고 이 모델을 test 데이터에 적용하여 최종적으로 test 데이터의 승객들의 생존여부를 예측하면 됩니다. 그 후 test 데이터 (승객 칼럼, Survived 칼럼만 나오게) 를 kaggle에 submit하면 learderboard에 score에서 정확도를 알 수 있습니다.  

## 데이터 소개  

|변수|설명|key|  
|:---|:---|:---|  
|survival|생존 여부|0 = 사망, 1 = 생존|  
|pclass|티켓 클래스|1 = 1등석, 2 = 2등석, 3 = 3등석|  
|sex|성별|1 = 남자, 2 = 여자|  
|Age|<u>나이</u>||  
|sibsp|<u>가족관계</u>||  
|parch|<u>가족관계</u>||  
|ticket|티켓 넘버||  
|fare|여객 요금||  
|carbin|객실번호||  
|embarked|승선했던 곳|C = Cherbourg, Q = Qyeebstown, S = Southampton|  

Age : 나이가 1보다 작으면 (xx.5 형태) 추정되었다고 봅니다.  
sibsp : 형제, 자매, 남편, 아내와 같은 형태입니다.  
parch : 부모 자녀 같은 형태입니다.  

<br/><br/>

## 전처리
프로젝트를 진행하기 앞서 데이터는 양적 데이터를 학습시킵니다. 그리고 많은 데이터를 학습시킨다면 더 좋은 성능의 모델이 될것입니다. 하지만 우리에게 주어진 데이터에는 양적 데이터와 질적데이터가 섞여 있어 우리는 질적데이터와 우리가 원하는 답인 생존률의 관계를 비교하여 연관성이 높은 질적데이터가 있다면 이 데이터를 양적데이터로 변환시키는 것이 좋습니다.  

## 모델링

## 결과