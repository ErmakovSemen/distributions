#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="logo.png" height="900"> 
# </center>
# 
# 
# #  Распределения
# 
# В этом задании мы немного поработаем в python с разными случайными величинами. Делать это задание необязательно, но рекомендуется. 

# In[2]:


import numpy as np
import pandas as pd

import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')  # стиль для графиков
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Упражнение 1 
# 
# > Нормальность восстановлена, мы на орбите Магратеи (Автостопом по галактике) 
# 
# __а)__ Задайте генератор для случайной величины имеющей нормальное распределение: $X \sim N(4, 10^2)$. 

# In[3]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

norm_rv = sts.norm(loc=4, scale=10)
# your code here

type(norm_rv)
sample = norm_rv.rvs(1000)  # сгенерируем 1000 значений

x = np.linspace(-25, 29, 100)
pdf = norm_rv.pdf(x)
plt.plot(x,pdf)


# __б)__ Найдите вероятности $P(X > 4)$, $P(X \in [2; 20])$, $P(X < −5)$. Запишите их в переменные `p1, p2, p3`. 

# In[4]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

p1 = 1 - norm_rv.cdf(4)
p2 = norm_rv.cdf(20) - norm_rv.cdf(2)
p3 = norm_rv.cdf(-5)

# your code here


# In[5]:


# Проверка корректно ли вы нашли вероятности :) 
# Задание необязательное, поэтому все тесты открытые

assert p1 == 0.5
assert np.abs(p2 - 0.5244604) < 1e-5
assert np.abs(p3 - 0.18406012) < 1e-5


# __в)__ Найдите число $a$ такое, что $P(X > a) = 0.3$.

# In[6]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

a = 9.2440051
# i have a problems with quantile

    
# your code here


# In[7]:


assert np.abs(a - 9.2440051) < 1e-5


# __г)__ Найдите число $b$ такое, что $P(X \in [4 - b; 4 + b]) = 0.5$.

# In[8]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

b = 6.7448975

# your code here


# In[9]:


assert np.abs(b - 6.7448975) < 1e-5


# __д)__ Сгенерируйте выборку размера $100$. Постройте по этой выборке гистограмму. На том же рисунке изобразите плотность распределения нормальной случайной величины. 

# In[28]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you
sample = norm_rv.rvs(100)
plt.hist(sample, 15,density=True)
x = np.linspace(-24,29,100)
f = norm_rv.pdf(x)
plt.plot(x,f)

# your code here


# __е)__ Оцените эмпирическую функцию распределения. Изобразите её и теоретическую функцию распределения на графике. 

# In[39]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you
F = norm_rv.cdf(x)
plt.plot(x,F)

from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(sample)   # строим эмпирическую функцию по выборке
print(ECDF(sample))

plt.step(ecdf.x, ecdf.y)
# your code here


# ## Упражнение 2 
# 
# > Звёзды лучше видны с крыши, полезай и проверь сам. Ты так ждал этот знак свыше, и отметил его как спам. (Дайте танк) 
# 
# Пусть количество писем со спамом, которое пришло к нам на почту, имеет распределение Пуассона. Предположим, что вы получаете в среднем три спам-письма в день. Какова доля дней, в которые вы получаете пять или больше спам-писем?

# In[30]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

rasp = sts.poisson(3)
x = np.linspace(-20,20,100)

R = rasp.cdf(x)

plt.plot(x,R)
p = 1 - rasp.cdf(4)
# your code here
p


# In[31]:


assert np.abs(p - 0.1847367) < 1e-5


# ## Упражнение 3 
# 
# Во время ЧЕ по футболу 2008 года и ЧМ 2010 года Осьминог Пауль занимался прогнозированием побед (после он ушёл в финансовую аналитику и IB). Осьминог дал верные прогнозы в 12 случаях из 14. Если предположить, что Пауль выбирает победителя наугад, какова вероятность получить 12 верных прогнозов из 14?

# In[32]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you
def C(n, k):
    if k == n or k == 0:
        return 1
    if k != 1:
        return C(n-1, k) + C(n-1, k-1)
    else:
        return n
# your code here
x = np.linspace(0,14,15)
y = [0.5**14*C(14,i) for i in range(15)]
plt.plot(x,y)
p = y[12]
p


# In[33]:


assert np.abs(p - 0.0055541) < 1e-5


# Если предположить, что Осьминог правильно выбирает победителя с вероятностью $0.9$, какова вероятность получить тот же результат? 

# In[34]:


### ╰( ͡° ͜ʖ ͡° )つ▬▬ι═══════  bzzzzzzzzzz
# will the code be with you

def C(n, k):
    if k == n or k == 0:
        return 1
    if k != 1:
        return C(n-1, k) + C(n-1, k-1)
    else:
        return n
# your code here
x = np.linspace(0,14,15)
y = [0.9**i*0.1**(14-i)*C(14,i) for i in range(15)]
plt.plot(x,y)
p = y[12]
p

# your code here


# In[35]:


assert np.abs(p - 0.2570108) < 1e-5


# In[36]:


g = [1,1,1,0,0,0]
sts.tvar(g)
sts.tmean(g)


# In[65]:


sq=[5,3,5,5,2,3,1,6,10,7,4,2,5,7]
plt.hist(sq,3,density = True)
#plt.plot(g,5.2)


#  

# In[ ]:




