# coding: utf-8

#-------------------General memo---------------------------------------------
#よくやる標準入力受け取り
import sys
DATA = sys.stdin.read()
DATA = DATA.replace("\n","")
DATA = DATA.split()

#PrintのFormating
print("{0} is {1} years old and he is {2}".format("Mike",35,"scary"))
print("a: %f, b: %f" %(a , b))

#改行なし Print
print(text,end='')

#Enumerate
x=["item1","item2","item3"]
for index,item in enumerate(x):
    print(index,":",item)

y={'key1':'val1', 'key2':'val2', 'key3':'val3'}
for index,key in enumerate(y):
    print("{0} : {1} : {2}".format(index,key,y[key]))

#Function 引数の個数指定なし
def biggest_number(*args):
    return max(args)
X = biggest_number(-10, -5, 5, 10)

#Counter
from collections import Counter
c = Counter([1,1,2,3,3,4,5,5,5])
print(c.most_common(3))

#Datetime
import datetime                       
everything = dir(datetime)         
print(everything)

#-------------------List 関連---------------------------------------------
my_list = list(range(1, 11))
backwards = my_list[::-1]

l_data = ["score","40","50","80"]
tmp = l_data.pop(0)
n_total = sum(map(lambda x: int(x), l_data))

evens_to_50 = [x for x in range(51) if x % 2 == 0]
even_squares = [x**2 for x in range(1,11) if x % 2 == 0]
threes_and_fives = [x for x in range(1,16) if x %3 == 0 or x % 5 == 0]

my_list = list(filter(lambda x: x % 3 == 0,range(16)))
squares = list(filter(lambda x:x>=30 and x<= 70,[x**2 for x in range(1,11)]))

li = [3, 4, 3, 2, 5, 4]
li_unique = list(set(li))

l = [1,1,2,3,4,4,5,10]
if all(x < 10 for x in l):
    print("All numbers are less than 10")
if any(x >= 10 for x in l):
    print("There is a number greater than 10")

animals = ["aardvark", "badger", "duck", "emu", "fennec fox"]
duck_index = animals.index("duck")
animals.insert(duck_index,"cobra")

list_a = [3, 9, 17, 15, 19]
list_b = [2, 4, 8, 10, 30, 40, 50, 60, 70, 80, 90]
for a, b in zip(list_a, list_b):
    print(max(a,b), end="  ")

#-------------------Set 関連---------------------------------------------
set_a = {7,9,2}
set_b = {9,22,2,25}
set_c = {9,18,21,5}

set_b.add(12)
set_b.discard(0)

set_union = set_a.union(set_b)
set_intersection = set_a.intersection(set_b,set_c)
set_differences = set_a.difference(set_b,set_c)
set_symmetric_difference = set_a.symmetric_difference(set_b)

#-------------------Dictionary 関連---------------------------------------------
my_dict = {"food":"apple", "tool":"spoon", "person":"jay"}
print(my_dict.items())
print(my_dict.keys())
print(my_dict.values())

candidates = {"a":15, "b":25, "c":35}
winnerVote = int(max(candidates.values()))

#Sort
orig = {"1": 3,  "3": 1, "2": 2}
sortByKey = sorted(orig.items())
sortByVal = sorted(orig.items(), key=lambda x:x[1])

#-------------------Str 関連---------------------------------------------
str = "abcde"
L = list(str)
str2 = "".join(L)

phrase = "A bird in the hand..."
for c in phrase:
    if c.lower() == 'a':
        print('X',end='')
    else:
        print(c,end='')

#-------------------Binary 関連---------------------------------------------
x = 0b1001 + 0b1001
y = 0b1001 * 0b1001
print (int("111",2))
print (int("0b100",2))
print (bin(5))
print (int(bin(5),2))

#-------------------JSON入出力---------------------------------------------
import json
dic = {'key1':'val1', 'key2':'val2', 'key3':'val3', 'key4':'val4'}
f = open('output.json', 'w')
json.dump(dic, f)
f.close()

f = open('output.json', 'r')
dic = json.load(f)
f.close()

#-------------------ファイル入出力---------------------------------------------
#ファイル入出力
item = "content"
f = open("output.txt", "w")
f.write(item + "\n")
f.close()

#Requests ------------------------------------------------------------------------------------------------
import requests
import pandas as pd
git_res = requests.get('https://api.github.com/search/repositories?q=language:python+created:2017-07-28&per_page=3')
outP = pd.DataFrame(git_res.json()['items'])[:][['language','stargazers_count','git_url','updated_at','created_at']]
#print(git_res.text)
print(outP)

# Class 使用例------------------------------------------------------------------------------------------------------------
class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age
    def __repr__(self):
        return repr((self.name, self.grade, self.age))
l_obj_students = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]

# ageでsort
print(sorted(l_obj_students, key=attrgetter('age'))) 

# gradeでsort さらにageでsort
print(sorted(l_obj_students, key=attrgetter('grade', 'age')))


# Class 使用例 2 ------------------------------------------------------------------------------------------------------------
class Animal(object):
    is_alive = True
    num_Animals = 0
    AnimalDict={}
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Animal.num_Animals += 1
        Animal.AnimalDict[name] = self
        
zebra = Animal("Jeffrey", 2)
giraffe = Animal("Bruce", 1)
panda = Animal("Chad", 7)

print(zebra.name, zebra.age, zebra.is_alive)
print(giraffe.name, giraffe.age, giraffe.is_alive)
print(panda.name, panda.age, panda.is_alive)
print(Animal.num_Animals)
print(Animal.AnimalDict)
print(Animal.AnimalDict[""Bruce""].age)"

# Class 使用例 3 承継 -------------------------------------------------------------------------------------------

class Shape(object):
    def __init__(self, number_of_sides):
        self.number_of_sides = number_of_sides
    def __repr__(self):
        x = ""I have "" + str(self.number_of_sides) + "" sides""
        return x

class Triangle(Shape):
    def __init__(self,side1, side2, side3):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
    def __repr__(self):
        x = ""I have "" + str(3) + "" sides""
        return x

myShape = Triangle(1,2,3)
print(myShape.side1)
print(myShape)

yourShape = Shape(4)
print(yourShape)"

#-------------------Pandas General memo---------------------------------------------
import pandas as pd

df = pd.read_csv('train.csv')
df.to_csv('modified.csv')

df_temp = df.copy() #こうしないと参照コピーになる

df.describe()
df.info()
df.isnull().sum()
df.corr()

df = df.fillna(0)
df = pd.concat((df, df_x), axis = 1)
df = df.dropna()
df = df.drop('ColumnName',axis = 1)

df_x = pd.Series(l_x, name = "ColumnName")
l_names = df.Name.values.tolist()

n_oldest = df.Age.max()
df_x = df[df.Age==n_oldest]
df_x = df[df.Age>=40]
n_family = df.Family.apply(lambda x: (df.Family == x).sum())

a = df.Sex == 'male'
a.sum()

cabin_known_ratio = len(df[((df.Cabin.isnull()==False) & (df.Survived == 1))]) / len(df[df.Cabin.isnull()==False])
cabin_unknown_ratio = len(df[((df.Cabin.isnull()) & (df.Survived == 1))]) / len(df[df.Cabin.isnull()])

#-------------------Pandas ダミー変数化
def dummylize(df,n_item):
    dum = pd.get_dummies(df[n_item], drop_first = True)
    df = pd.concat((df, dum),axis = 1)
    df = df.drop(n_item,axis = 1)
    return df

#-------------------Pandas カテゴリをOne-Hot化
y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),10)

#-------------------Pandas nullを0、それ以外を1とする
def booleanize(df,n_item):
    df_tmp = df[n_item].notnull() * 1
    df_tmp.name = n_item

    df = df.drop(n_item,axis = 1)
    df = pd.concat((df, df_tmp), axis = 1)
    return df

#-------------------Pandas カテゴリを数値化 2
def categorize(df,n_item):
    labels, uniques = pd.factorize(df[n_item])
    df[n_item] = labels
    return df

#-------------------Pandas カテゴリを数値化 2 ["Miss","Mr","Mrs","Master","Miss"] → [0,1,2,3,0]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
num_cat = le.fit_transform(str_categories)
df_n_cat = pd.Series(num_cat, name='NumCat')


#-------------------Pandas 列加工もろもろ
choice = ["Miss","Mr","Mrs","Master"]
l_title = [list(set(x).intersection(set(choice)))[0] for x in l_names]     # ["Mr","Johnny","Depp"] -> "Mr"

def ageGroup(x):
        if 0 <= x and x < 10:
            return 0
        elif 10 <= x and x < 20:
            return 1
        elif 20 <= x and x < 40:
            return 2
        elif 40 <= x and x < 60:
            return 3
        elif 60 <= x:
            return 4
df_train.Age = df_train.Age.apply(lambda x: ageGroup(x))

#-------------------Pandas cut
pd.cut(df, [0, 10, 50, 100])
pd.cut(df, 4, right=False)
counts = pd.cut(df, 3, labels=['S', 'M', 'L']).value_counts()
s_cut, bins = pd.cut(df, 4, retbins=True)
print(s_cut)
print(bins)

#-------------------Pandas NaN 部分のみを予測モデルで補完 
    df_res_pred = model.predict(X)
    df_real_age = df['Age'].fillna(0)
    l_mix = []
    for i,v in enumerate(df_real_age):
        if v != 0:
            l_mix.append(v)
        else:
            l_mix.append(df_res_pred[i])

    df_mix = pd.Series(l_mix, name='Age')
    df = df.drop('Age',axis = 1)
    df = pd.concat((df,df_mix),axis = 1)

#-------------------Matplot General memo---------------------------------------------
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df_draw = df_tmp[df_tmp.Survived==1]
df_age = df_draw.iloc[:,2]
df_sex = df_draw.iloc[:,6]
plt.scatter(df_age,df_sex, color='#cc6699',alpha=0.5)
df_draw = df_tmp[df_tmp.Survived==0]
df_age = df_draw.iloc[:,2]
df_sex = df_draw.iloc[:,6]
plt.scatter(df_age,df_sex, color='#6699cc',alpha=0.5)
plt.show()

sex_survived_ratio = [male_survived_ratio, female_survived_ratio]
plt.bar([0,1], sex_survived_ratio, tick_label=['male', 'female'], width=0.5)
plt.show()

#機械学習関連メモ General --------------------------------------------------------------------------------
#モデル保存
import pickle
filename = 'xgb_model.sav'
pickle.dump(model_final, open(filename, 'wb'))

#モデル読み込み
import pickle
filename = 'xgb_model.sav'
loaded_m_xgb = pickle.load(open(filename, 'rb'))

#学習・テスト用データセット整備
from sklearn.model_selection import train_test_split
df_train = df_train.fillna(0)
l_features = ['Pclass','male','Q','S','TitlesNum','FamilyNum','Cabin','Age','SibSp','Parch','Fare']
df_x = df_train[l_features]
df_y = df_train.Survived
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25)

#model 比較  --------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
models =[]
models.append(model_xgb)
models.append(model_randForest)
models.append(model_MLPC)
for model in models:    
    res_tmp = model.predict(X_test)
    res_mean = mean_squared_error(y_test, res_tmp)
    print('mean squared error:{0} '.format(res_mean))

#Grid Search Cross Validation-----------------------------------------------------------------------------------
def applyGSCV(model, param, X, Y):
    from sklearn.model_selection import GridSearchCV
    res = GridSearchCV(model, param, cv=3)
    res.fit(X, Y)
    return res
from sklearn.ensemble import RandomForestClassifier
model_selected = RandomForestClassifier()
obj_param = {
'n_estimators': [5,10,20,30,50,100,300],
'max_depth': [3,5,10,15,20,25,30,40,50,100],
'random_state': [0]
}
model_final = applyGSCV(model, obj_param, df_x, df_y)

#XGBoostモデル--------------------------------------------------------------------------------
def xgb_model(X, Y):
    import xgboost as xgb
    import scipy.stats as st
    from sklearn.model_selection import RandomizedSearchCV
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    params = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }
    xgbreg = xgb.XGBRegressor(nthreads=-1)
    gs = RandomizedSearchCV(xgbreg, params, n_jobs=3)
    res = gs.fit(X, Y)
    print(gs.best_params_)
    return res
model_xgb = xgb_model(X_train, y_train)

#Random Forestモデル--------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
obj_param = {
    'n_estimators': [5,10,20,30,50,100,300], 'max_depth': [3,5,10,15,20,25,30,40,50,100], 'random_state': [0]
}
model_randForest = applyGSCV(RandomForestClassifier(),obj_param,X_train, y_train)

#MLPCモデル--------------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
model_MLPC =  MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)

#Keras NNモデル---------------------------------------------------------------------------------------
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical

N = len(l_features)
model_NN = Sequential()
model_NN.add(Dense(2,input_dim=N, activation='sigmoid', kernel_initializer='uniform'))
model_NN.add(Dense(2,activation='softmax', kernel_initializer='uniform'))
sgd = keras.optimizers.SGD(lr = 0.5, momentum = 0.0,decay = 0.0, nesterov = False)
model_NN.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

y_train_bin = to_categorical(y_train)
y_test_bin = to_categorical(y_test)

history = model_NN.fit(X_train, y_train_bin, batch_size=100,epochs=1000,verbose=0,validation_data=(X_test, y_test_bin))
history.history['acc'] #対trainデータ正確性
history.history['val_acc'] #対testデータ正確性

score = model_NN.evaluate(X_test, y_test_bin, verbose = 0) 
print(score[0], score[1]) #score[0]が交差エントロピー誤差、score[1]がテストデータ正答率

#複数のモデルをざっくり試す----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
names = ['LogisticRegression', 'SVC', 'LinearSVC', 'KNeighbors', 'DecisionTree', 'RandomForest', 'MLPClassifier']
l_models = []
l_models.append(("LogisticRegression", LogisticRegression()))
l_models.append(("SVC", SVC()))
l_models.append(("LinearSVC", LinearSVC()))
l_models.append(("KNeighbors", KNeighborsClassifier()))
l_models.append(("DecisionTree", DecisionTreeClassifier()))
l_models.append(("RandomForest", RandomForestClassifier()))
l_models.append(("MLPClassifier", MLPClassifier(solver='lbfgs', random_state=0)))
def evaluate_models(df_forAgePred, l_pred, l_models):
    results = []
    names = []
    X_train, X_test, y_train, y_test = train_test_split(df_forAgePred[l_pred], df_forAgePred.Age, test_size=0.25)
    for name, model in l_models:
        model.fit(X_train, y_train)
        res_pred = model.predict(X_test)
        result = mean_squared_error(y_test, res_pred)
        names.append(name)
        results.append(result)
    return names, results
import statistics
result_list = []
for i in range(0,10):
    print('test {0}'.format(i))
    name, res = evaluate_models(df_forAgePred, l_pred, l_models)
    for x, y in zip(name, res):
        result_list.append([x, y])
print('mean squared error results by model')
for n in names:
    r = [i[1] for i in result_list if i[0] == n]
    print(n)
    print('avg: {0:,.2f} / median: {1:,.2f}'.format(sum(r)/len(r), statistics.median(r)))

#Tensorflow モデル保存 & 読み込み--------------------------------------------------------------------------
#Save
saver = tf.train.Saver()
saver.save(sess, '../model/test_model')
#Read
saver = tf.train.Saver()
saver.restore(sess, '../model/test_model')

#Tensorflow 使用例 1 --------------------------------------------------------------------------
import tensorflow as tf
# Model parameters (variables of function to be learned via machine learning)
W = tf.Variable([0.], dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)
# Model input and output (sets of data and result)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# Model itself
linear_model = W*x + b
# loss
deltas = linear_model - y
square_deltas = tf.square(deltas)
loss = tf.reduce_sum(square_deltas) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_dataset = [1, 2, 3, 4]
y_dataset = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 
for x in range(1000):
  sess.run(train, {x: x_dataset, y: y_dataset})
    if (x+1) % 100 == 0:
        print('\nStep: %s' % (x+1))
        print('loss: ' + str(sess.run(train, {x: x_dataset, y: y_dataset})))
        print("W: %f, b: %f" % (sess.run( W ), sess.run( b )) )
# evaluate training accuracy
ans_W, ans_b, ans_loss = sess.run([W, b, loss], {x: x_dataset, y: y_dataset})
print("W: %s b: %s loss: %s"%(ans_W, ans_b, ans_loss))

# Tensorflow 使用例 2 ------------------------------------------------------------------

import numpy as np
import tensorflow as tf

num_columns = [tf.feature_column.numeric_column("x",shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=num_columns)

x_dataset = np.array([1.,2.,3.,4.])
y_dataset = np.array([0.,-1.,-2.,-3.])

x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01,-4.1,-7.,0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_dataset},y_dataset,batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_dataset},y_dataset,batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_eval},y_eval,batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

ans_train_data = estimator.evaluate(input_fn=train_input_fn)
ans_eval_data = estimator.evaluate(input_fn=eval_input_fn)

print("train data vs model: %r" % ans_train_data)
print("eval data vs model: %r" % ans_eval_data)


# Tensorflow 使用例 3 ------------------------------------------------------------------

import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))

  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),tf.assign_add(global_step, 1))

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)


estimator = tf.estimator.Estimator(model_fn=model_fn)

x_train_dataset = np.array([1., 2., 3., 4.])
y_train_dataset = np.array([0., -1., -2., -3.])

x_eval_dataset = np.array([2., 5., 8., 1.])
y_eval_dataset = np.array([-1.01, -4.1, -7., 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train_dataset}, y_train_dataset, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train_dataset}, y_train_dataset, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval_dataset}, y_eval_dataset, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)


train_results = estimator.evaluate(input_fn=train_input_fn)
eval_results = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_results)
print("eval metrics: %r"% eval_results)


# tensorflow MNIST example -------------------------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
import tensorflow as tf

#variables
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#model formula
y = tf.nn.softmax(tf.matmul(x, W) + b)

#cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#run
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#accuracy check
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))



#Keras mnist 使用例----------------------------------------------------------------------------------


# import libraries
import pandas as pd
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Sequential Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

# Download MNIST datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show sample data
fig = plt.figure(figsize=(9,9))
for i in range(36):
    ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gist_gray')
plt.show()

# reshape 28*28 pixel data into 784 dim data
# convert into float type and normalize pixel data from 0.0 to 1.0
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

# encode label data into "one-hot"
y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),10)
y_test = keras.utils.np_utils.to_categorical(y_test.astype('int32'),10)

# select Sequiential model
model = Sequential()

# 1st layer : fully connected layer(output:512)
# only first layer needs to define input_shape
model.add(Dense(512, activation='relu', input_shape=(784,)))

# use Dropout regularization rate to avoid overfitting
# Randomly ignoring connections between layers
model.add(Dropout(0.2))

# 2nd layer : fully connected layer(output:512)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# 3rd layer : fully connected layer(output:10)
# acrivation methods: softmax, which squashes the outputs of each unit to be between 0 and 1.(often used in the final layer)
model.add(Dense(10, activation='softmax'))

# Set definitions for traning
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Excute training for 20(epochs) times
history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

# plot the resulut
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Keras mnist 使用例 2 ----------------------------------------------------------------------------------

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 3            #original 12

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Kerasのバックエンドで動くTensorFlowとTheanoでは入力チャンネルの順番が違うので場合分けして書いています
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))


