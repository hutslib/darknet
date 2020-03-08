#!user/bin/python
# _*_ coding: utf-8 _*_\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lines =80200
# for x in range(lines):
#     print x%10
#     if x%10!=10:
#         print x
result = pd.read_csv('/home/hts/darknet/person_train_log_loss.txt', skiprows=[x for x in range(lines) if ((x%10!=9) |(x<30000)| (x>40000))]  ,
                    error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
#print result
result.head()#输出前五行(默认的n=5)

#print 'finish'
#print result['loss']
#print result['images'].str.split(' ').str.get(2)

result['loss']=result['loss'].str.split(' ').str.get(1)
result['avg']=result['avg'].str.split(' ').str.get(1)
result['rate']=result['rate'].str.split(' ').str.get(1)
result['seconds']=result['seconds'].str.split(' ').str.get(1)
result['images']=result['images'].str.split(' ').str.get(1)
result.head()
result.tail()

#print(result.head())
# print(result.tail())
# print(result.dtypes)

print(result['loss'])
print(result['avg'])
print(result['rate'])
print(result['seconds'])
print(result['images'])

result['loss']=pd.to_numeric(result['loss'])
result['avg']=pd.to_numeric(result['avg'])
result['rate']=pd.to_numeric(result['rate'])
result['seconds']=pd.to_numeric(result['seconds'])
result['images']=pd.to_numeric(result['images'])
result.dtypes


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #111”表示“1×1网，第一子图”,“234”表示“2×3网格，第四子图”
ax.plot(result['avg'].values,label='avg_loss')
#ax.plot(result['loss'].values,label='loss')
ax.legend(loc='best')
ax.set_title('The loss curves')
ax.set_xlabel('batches')
fig.savefig('avg_loss')
#fig.savefig('loss')
