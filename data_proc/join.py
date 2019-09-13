import pandas as pd


f1=pd.read_csv('MVI_3255_sync.csv')
f2=pd.read_csv('MVI_3256_sync.csv')
f3=pd.read_csv('MVI_3257_sync.csv')
f4=pd.read_csv('MVI_3258_sync.csv')


vids=[f1,f2,f3,f4]

listener=pd.concat(vids)
print(listener.shape)
pd.DataFrame(listener).to_csv('listener.csv')