import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


lines =80200
result = pd.read_csv('/home/hts/darknet/person_train_log_iou.txt', skiprows=[x for x in range(lines) if (x%10!=9)] ,error_bad_lines=False, names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall','count'])
result.head()

result['Region Avg IOU']=result['Region Avg IOU'].str.split(': ').str.get(1)
result['Class']=result['Class'].str.split(': ').str.get(1)
result['Obj']=result['Obj'].str.split(': ').str.get(1)
result['No Obj']=result['No Obj'].str.split(': ').str.get(1)
result['Avg Recall']=result['Avg Recall'].str.split(': ').str.get(1)
result['count']=result['count'].str.split(': ').str.get(1)
result.head()
result.tail()

#print(result.head())
# print(result.tail())
# print(result.dtypes)
print(result['Region Avg IOU'])

result['Region Avg IOU']=pd.to_numeric(result['Region Avg IOU'])
result['Class']=pd.to_numeric(result['Class'])
result['Obj']=pd.to_numeric(result['Obj'])
result['No Obj']=pd.to_numeric(result['No Obj'])
result['Avg Recall']=pd.to_numeric(result['Avg Recall'])
result['count']=pd.to_numeric(result['count'])
result.dtypes
#绘制IOU
fig_IOU = plt.figure()
ax_IOU = fig.add_subplot(1, 1, 1)
ax_IOU.plot(result['Avg IOU'].values,label='Avg IOU')
ax_IOU.legend(loc='best')
ax_IOU.set_title('The Avg IOU curves')
ax_IOU.set_xlabel('batches')
fig_IOU.savefig('Region Avg IOU')

#绘制Avg Recall
fig_IOU = plt.figure()
ax_IOU = fig.add_subplot(1, 1, 1)
ax_IOU.plot(result['Avg Recall'].values,label=''Avg Recall'')
ax_IOU.legend(loc='best')
ax_IOU.set_title('The 'Avg Recall' curves')
ax_IOU.set_xlabel('batches')
fig_IOU.savefig(''Avg Recall'')

# ax.plot(result['Class'].values,label='Class')
# ax.plot(result['Obj'].values,label='Obj')
# ax.plot(result['No Obj'].values,label='No Obj')
# ax.plot(result['Avg Recall'].values,label='Avg Recall')
# ax.plot(result['count'].values,label='count')
#ax.set_title('The Region Avg IOU curves')
#fig.savefig('Avg IOU')

