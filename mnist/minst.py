##利用keras 来搭建一个网络来
import pandas as pd
from keras.models import Sequential
##准备数据
train = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_label,train_data = train.values[:,0],train.values[:,1:]
model = Sequential()
from keras.layers import Dense,Activation
model.add(Dense(units=64,input_shape=(None,784)))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
##编译模型
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=['accuracy'])
#训练数据
print("开始训练模型")
model.fit(train_data,train_label,epochs=5,batch_size=32)
print("模型训练结束,开始预测数据")
classes = model.predict(test_data,batch_size=128)
print("预测数据结束")
print(classes)