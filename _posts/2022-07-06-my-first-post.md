---
title:  "첫번째 포스팅"
excerpt: "첫번째 포스팅이다. 이것저것 실험해보고 더 나은 블로그를 위하여 발전시켜 보자! "

categories:
  - Blog
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
toc_label : "C O N T E N T S"
 
date: 2022-07-07
last_modified_at: 2022-07-07
---

# 마크다운 미리 확인하기
visual studio code에서 오른쪽 위 메뉴를 보면 측면에서 미리보기 열기를 클릭하여 미리 확인을 할 수 있다.
한번 더 확인 해보기!!  


# Liquid 문법 간단히 알아보기

## 1. Object   
`{{`와 `}}`을 사용하여 감싼다. page에 `{{` `}}`로 감싸져 있는 Object를 출력한다. (변수라고 생각하면 됨.)  
{{ page.title }}  


## 2. Tags  
`{{%` 와 `%}}`을 사용하여 감싼다. 논리와 제어를 담당하는 역할을 한다.  
### 1. Control flow (조건문)  
if 문 : `{% if 조건문 %} ... {% endif %}`  
if-else if-else 문 : `{% if 조건문 %}{% elsif 조건문 %}{% else 조건문 %} ... {% endif %}`  
if not 문 : `{% unless 조건문 %} ... {% endunless %}`  
switch-case 문 : `{% case 조건문 %}{% when 값 %} ... {% endcase %}`  


### 2. Iteration (반복문)  
### for 문  

```
{% raw %}  
{% for product in collection.products %}  
  {{ product.title }}  
{% endfor %}  
{% endraw %}
```

신기하게 liquid 문법은 for 문에서도 `else`를 쓸 수 있다. for문에서의 `else`는 반복하려는 컨텐츠가 비어서 for 문을 한번도 돌릴 수 없을 때 해당된다.  

```
{% raw %}
{% for product in collection.products %}
  {{ product.title }}
{% else %}
  The collection is empty.
{% endfor %}
{% endraw %}
```  

이밖에도 for문에 `limit`, `offset`, `range` 등등으로 반복 횟수와 범위를 제어할 수 있다.  
  
### cycle 문  
계속 순환하기 때문에 `"second" cycle`이 네번째 순회할 땐 다시 one을 출력하게 된다.  

```
{% raw %}
{% cycle "first" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "first" : "one", "two", "three" %}
{% endraw %}
```  
{% cycle "first" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "second" : "one", "two", "three" %}
{% cycle "first" : "one", "two", "three" %}  

### 3. Variable assignments  
### assign  
변수를 새로 만들고 할당함  

```
{% raw %}
{% assign foo = "bar" %}
{{ foo }}
{% endraw %}
```
{% assign foo = "bar" %}
{{ foo }}

### capture  
태그로 둘러쌓여 있는 문자열을 변수에 저장한다.  

```
{% raw %}
{% capture my_variable %}I am being captured.{% endcapture %}
{{ my_variable }}
{% endraw %}
```
{% capture my_variable %}I am being captured.{% endcapture %}
{{ my_variable }}  

### increment, decrement  
변수의 값을 증가하고 감소시킨다.  
  
## Liquid 문법 그대로를 출력하고 싶을 때  
liquid 문법을 `{% raw %}``{% endraw %}`로 감싼다.  

### 주석  
```
{% raw %}
Anyting you put between {% comment %} and {% endcomment %} tags
is turned into a commet
{% endraw %}
``` 
Anyting you put between {% comment %} and {% endcomment %} tags
is turned into a commet  

### 줄 바꿈 하고 싶지 않을 때  
텍스트를 출력하지 않더라도 Liquid 언어 상 태그를 사용하면 한 줄이 출려고딜 수 있다. 공백 한 줄 출력되고 싶지 않다면 % 안쪽에 `-`를 붙여주자.  

```
{% raw %}
{%- assign my_variable = "tomato" -%}
{{ my_variable }}
{% endraw %}
```
{%- assign my_variable = "tomato" -%}
{{ my_variable }}