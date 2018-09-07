#-------------------General memo---------------------------------------------
# coding: utf-8
#よくやる標準入力受け取り
import sys
DATA = sys.stdin.read()
DATA = DATA.replace("\n","")
DATA = DATA.split()

#PrintのFormating
print("{0} is {1} years old and he is {2}".format("Mike",35,"scary"))



#List処理関連
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

#Dictionary関連
my_dict = {"food":"apple", "tool":"spoon", "person":"jay"}
print(my_dict.items())
print(my_dict.keys())
print(my_dict.values())

candidates = {"a":15, "b":25, "c":35}
winnerVote = int(max(candidates.values()))

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

#Sort関連
orig = {"1": 3,  "3": 1, "2": 2}
sortByKey = sorted(orig.items())
sortByVal = sorted(orig.items(), key=lambda x:x[1])

#Set関連
set_a = {7,9,2}
set_b = {9,22,2,25}
set_c = {9,18,21,5}

set_b.add(12)
set_b.discard(0)

set_union = set_a.union(set_b)
set_intersection = set_a.intersection(set_b,set_c)
set_difference = set_a.difference(set_b)
set_differences = set_a.difference(set_b,set_c)
set_difference_1 = set_a.difference(set_b)
set_difference_2 = set_b.difference(set_a)
set_symmetric_difference = set_a.symmetric_difference(set_b)

#Str関連
str = "abcde"
L = list(str)
str2 = "".join(L)

phrase = "A bird in the hand..."
for char in phrase:
    if char.lower() == 'a':
        print('X',end='')
    else:
        print(char,end='')

#Binary関連
x = 0b1001 + 0b1001
y = 0b1001 * 0b1001
print (int("111",2))
print (int("0b100",2))
print (bin(5))
print (int(bin(5),2))

#Counter
from collections import Counter
c = Counter([1,1,2,3,3,4,5,5,5])
print(c.most_common(3))

#Datetime
import datetime                       
everything = dir(datetime)         
print(everything)

#JSON関連
import json
dic = {'key1':'val1', 'key2':'val2', 'key3':'val3', 'key4':'val4'}
f = open('output.json', 'w')
json.dump(dic, f)
f.close()
f = open('output.json', 'r')
loaded_dic = json.load(f)
f.close()
print("loaded dictionary is: ", loaded_dic)

#ファイル入出力
item = "content"
f = open("output.txt", "w")
f.write(item + "\n")
f.close()


#Requests -------------------------------------------------------------------------------------------------------------

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

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]

# ageでsort
print(sorted(student_objects, key=attrgetter('age'))) 

# gradeでsort さらにageでsort
print(sorted(student_objects, key=attrgetter('grade', 'age')))


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

#-------------------Pandas memo---------------------------------------------

#pandasインポートとデータセット読み込み
import pandas as pd
df_train = pd.read_csv('train.csv')

#describe
df_train.describe()

#info
df_train.info()

#欠損値の数をカウント
df_train.isnull().sum()

#項目ごとの最大値を取得
n_oldest = df_train.Age.max()

#最大値をもつ行を取得する方法
df_train[df_train.Age==n_oldest]

#相関係数
df_tmp.corr()

#ダミー変数化処理用関数

def dummylize(df,n_item):
    dum = pd.get_dummies(df[n_item], drop_first = True)
    df = pd.concat((df, dum),axis = 1)
    df = df.drop(n_item,axis = 1)
    return df

#nullを0、それ以外を1とする関数
def booleanize(df,n_item):
    df_tmp = df[n_item].notnull() * 1
    df_tmp.name = n_item

    df = df.drop(n_item,axis = 1)
    df = pd.concat((df, df_tmp), axis = 1)
    return df

#カテゴリを数値化する関数

def categorize(df,n_item):
    labels, uniques = pd.factorize(df[n_item])
    df[n_item] = labels
    return df


#カテゴリをOne-Hot化
y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),10)


#条件で要素抜出し
df_x = df_train[df_train.Age>=40]

#項目のある値の出現頻度をカウント
a = df_train.Sex == 'male'
a.sum()

#欠損データの行削除
df_tmp = df_train.dropna()

#不要行削除
df_tmp = df_tmp.drop('PassengerId',axis = 1)
df_tmp = df_tmp.drop('Name',axis = 1)
df_tmp = df_tmp.drop('Ticket',axis = 1)
df_tmp = df_tmp.drop('Cabin',axis = 1)



#性別・年代別の生存率を散布図プロット

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


#キャビン番号が分かっているケースと不明のケースの生存率比較

cabin_known_ratio = len(df_train[((df_train.Cabin.isnull()==False) & (df_train.Survived == 1))]) / len(df_train[df_train.Cabin.isnull()==False])
cabin_unknown_ratio = len(df_train[((df_train.Cabin.isnull()) & (df_train.Survived == 1))]) / len(df_train[df_train.Cabin.isnull()])


#男女生存率を比較

male_survived_ratio = len(df_train[(df_train.Sex == 'male') & (df_train.Survived == 1)]) / len(df_train[df_train.Sex == 'male'])
female_survived_ratio = len(df_train[(df_train.Sex == 'female') & (df_train.Survived == 1)]) / len(df_train[df_train.Sex == 'female'])
sex_survived_ratio = [male_survived_ratio, female_survived_ratio]
plt.bar([0,1], sex_survived_ratio, tick_label=['male', 'female'], width=0.5)
plt.show()


#年代別　男女生存比率　比較チャート

male_df = df_train[df_train.Sex == 'male']
female_df = df_train[df_train.Sex == 'female']

male_age_survived_ratio_list = []
female_age_survived_ratio_list = []
for i in range(0, int(max(df_train.Age))+1, 10):
    male_df_of_age = male_df[(male_df.Age >= i) & (male_df.Age < i+9)]
    female_df_of_age = female_df[(female_df.Age >= i) & (female_df.Age < i+9)]

    male_s = len(male_df_of_age[male_df_of_age.Survived == 1])
    female_s = len(female_df_of_age[female_df_of_age.Survived == 1])

    male_total = len(male_df_of_age)
    female_total = len(female_df_of_age)

    if male_total  == 0:
        male_age_survived_ratio_list.append(0.5)
    else:
        male_age_survived_ratio_list.append(male_s/male_total)

    if female_total == 0:
        female_age_survived_ratio_list.append(0.5)
    else:
        female_age_survived_ratio_list.append(female_s/female_total)

print(male_age_survived_ratio_list, female_age_survived_ratio_list)

x_labels = []
for i in range(0, int(max(df_train.Age))+1, 10):
    x_labels.append(str(i) + '-' + str(i+9))

plt.figure(figsize=(16,8))
x1 = [i for i in range(0, int(max(df_train.Age))+ 1, 10)]
x2 = [i + 2 for i in range(0, int(max(df_train.Age))+ 1, 10)]
plt.bar(x1, male_age_survived_ratio_list, tick_label=x_labels, width=3, color='blue')
plt.bar(x2,female_age_survived_ratio_list, tick_label=x_labels, width=3, color='red')
plt.tick_params(labelsize = 15)
plt.show()


#ファミリーネーム抜出し

l_names = [x.replace(",","").replace(".","").split(" ") for x in df_train.Name.values.tolist()]
l_family = [x[0] for x in l_names]
df_family = pd.Series(l_family, name = "Family")

#ファミリーネームの列を追加

df_train = pd.concat((df_train, df_family), axis = 1)
df_train.head()

#ファミリーネームの出現回数列を追加

n_family = df_train.Family.apply(lambda x: (df_train.Family == x).sum())
n_family.name = 'FamilyNum'
df_train = pd.concat((df_train, n_family), axis = 1)

#家族の構成人数ごとの生存率比較

f1_ratio = len(df_train[((df_train.FamilyNum == 1) & (df_train.Survived == 1))]) / len(df_train[df_train.FamilyNum == 1])
f23_ratio = len(df_train[((df_train.FamilyNum >= 2) & (df_train.FamilyNum <= 3) & (df_train.Survived == 1))]) / len(df_train[((df_train.FamilyNum >= 2) & (df_train.FamilyNum <= 3))])
f4_ratio = len(df_train[((df_train.FamilyNum >= 4) & (df_train.Survived == 1))]) / len(df_train[df_train.FamilyNum >= 4])

l_ratios = []
l_ratios.append(f1_ratio)
l_ratios.append(f23_ratio)
l_ratios.append(f4_ratio)
l_ratios

plt.bar(['1','2-3','4+'],l_ratios)
plt.show()



#-----------------------------------------------------------------------------


#ダミー変数化処理用関数

def dummylize(df,n_item):
    dum = pd.get_dummies(df[n_item], drop_first = True)
    df = pd.concat((df, dum),axis = 1)
    df = df.drop(n_item,axis = 1)
    return df


#nullを0、それ以外を1とする関数
def booleanize(df,n_item):
    df_tmp = df[n_item].notnull() * 1
    df_tmp.name = n_item

    df = df.drop(n_item,axis = 1)
    df = pd.concat((df, df_tmp), axis = 1)
    return df


#Listを含まれるタイトルのみの文字列に変換するサブ関数

def titleCheck(L):
        if 'Miss' in L: return 'Miss'
        elif 'Mrs' in L: return 'Mrs'
        elif 'Master' in L: return 'Master'
        elif 'Mr' in L: return 'Mr'
        else: return ""


#処理関数メイン

def nameAnalysis(df):

    l_names = [x.replace(",","").replace(".","").split(" ") for x in df.Name.values.tolist()]
#Mr. Mrs. などを仕分け
    l_title = [titleCheck(x) for x in l_names]
    df_title = pd.Series(l_title, name='Titles')

#Mr. Mrs. などを数値化
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    num_title = le.fit_transform(l_title)
    df_n_title = pd.Series(num_title, name='TitlesNum')

#ファミリーネーム抜出し

    l_family = [x[0] for x in l_names]
    df_family = pd.Series(l_family, name = "Family")

#ファミリーネームの出現回数列を追加
    df_n_family = df_family.apply(lambda x: (df_family == x).sum())
    df_n_family.name = 'FamilyNum'

#作成した各分析列を追加しリターン
    df = pd.concat((df, df_title), axis = 1)
    df = pd.concat((df, df_n_title), axis = 1)
    df = pd.concat((df, df_family), axis = 1)
    df = pd.concat((df, df_n_family), axis = 1)
        
    return df


#年齢を世代区分け


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


#年齢推定モデルに渡すためテスト用加工済みCSVを保存

df_train.to_csv('forAgePred.csv')

#Age欠損部分を年齢推定モデルを使い補完する処理まとめ関数

def fillMissingAge(df):

    #モデルを使って予測し結果をリストに保存→Seriesに変換
    l_pred = ['TitlesNum']
    res_pred = model_age_pred.predict(df[l_pred])
    df_res_pred =pd.Series(res_pred, name='prediction')
    df_res_pred = df_res_pred.apply(lambda x: float(x/100))

    df_real_age = df['Age']
    df_real_age = df_real_age.fillna(0)


    #もともとのAge列の欠損値のみPredictの結果で置き換え　(もっといいやり方があるはず)

    df_real_age = df['Age']
    df_real_age = df_real_age.fillna(0)

    l_mix = []

    for i,v in enumerate(df_real_age):
        if v != 0:
            l_mix.append(v)
        else:
            l_mix.append(df_res_pred[i])

    df_mix = pd.Series(l_mix, name='Age')

    #Age列を差し替え
    df = df.drop('Age',axis = 1)
    df = pd.concat((df,df_mix),axis = 1)

    return df


# scikit-learn, 機械学習関連メモ--------------------------------------------------------------------------------

#別NoteBookの処理で作成・保存した年齢推定モデルの読み込み

import pickle
filename = 'age_pred.sav'
model_age_pred = pickle.load(open(filename, 'rb'))

#学習・テスト用データセット整備

from sklearn.model_selection import train_test_split
df_train = df_train.fillna(0)
l_features = ['Pclass','male','Q','S','TitlesNum','FamilyNum','Cabin','Age','SibSp','Parch','Fare']
df_x_pack = df_train[l_features]
df_y_pack = df_train.Survived
X_train, X_test, y_train, y_test = train_test_split(df_x_pack, df_y_pack, test_size=0.25)


#model 比較

from sklearn.metrics import mean_squared_error

models =[]
models.append(model_xgb)
models.append(model_randForest)
models.append(model_MLPC)

for model in models:    
    res_tmp = model.predict(X_test)
    res_mean = mean_squared_error(y_test, res_tmp)
    print('mean squared error:{0} '.format(res_mean))


#モデル保存

import pickle
filename = 'xgb_model.sav'
pickle.dump(model_final, open(filename, 'wb'))


#モデル読み込み

import pickle
filename = 'xgb_model.sav'
loaded_m_xgb = pickle.load(open(filename, 'rb'))


#予測結果を提出用CSVに整理

df_X_forSubmit = df_test[l_features]
Y_forSubmit = loaded_m_xgb.predict(df_X_forSubmit)
Y_forSubmit_int = [0 if x <=0.5 else 1 for x in Y_forSubmit]

df_Y = pd.Series(Y_forSubmit_int, name='Survived')
df_submit = df_test.PassengerId.copy()

df_submit = pd.concat((df_submit,df_Y),axis=1)

df_submit.head()
df_submit.to_csv('submission.csv', index = False)



#Grid Search Cross Validationでモデルを学習させる--------------------------------------------------------------

def applyGSCV(model, param, X, Y):
    from sklearn.model_selection import GridSearchCV

    res = GridSearchCV(model, param, cv=3)
    res.fit(X, Y)

    return res

#選択したモデルを学習させる

from sklearn.ensemble import RandomForestClassifier
model_selected = RandomForestClassifier()

obj_param = {
'n_estimators': [5,10,20,30,50,100,300],
'max_depth': [3,5,10,15,20,25,30,40,50,100],
'random_state': [0]
}

df_feature = df["familyNum","titleNum"]
df_answer = df["Age"]

model_final = applyGSCV(model_selected, obj_param, df_feature, df_answer)

print('done')


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

#XGBモデル学習

model_xgb = xgb_model(X_train, y_train)



#Random Forestモデル--------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

obj_param = {
'n_estimators': [5,10,20,30,50,100,300],
'max_depth': [3,5,10,15,20,25,30,40,50,100],
'random_state': [0]
}

model_randForest = applyGSCV(RandomForestClassifier(),obj_param,X_train, y_train)

print('done')

#MLPCモデル--------------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
model_MLPC =  MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)


#Keras NNモデル---------------------------------------------------------------------------------------

import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation

N = len(l_features)
model_NN = Sequential()
model_NN.add(Dense(2,input_dim=N, activation='sigmoid', kernel_initializer='uniform'))
model_NN.add(Dense(2,activation='softmax', kernel_initializer='uniform'))
sgd = keras.optimizers.SGD(lr = 0.5, momentum = 0.0,decay = 0.0, nesterov = False)
model_NN.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.utils import to_categorical

y_train_bin = to_categorical(y_train)
y_test_bin = to_categorical(y_test)


history = model_NN.fit(X_train, y_train_bin, batch_size=100,epochs=1000,verbose=0,validation_data=(X_test, y_test_bin))

#対trainデータ正確性
history.history['acc']

#対testデータ正確性
history.history['val_acc']

#score[0]が交差エントロピー誤差、score[1]がテストデータ正答率
score = model_NN.evaluate(X_test, y_test_bin, verbose = 0)
print(score[0], score[1])

#複数のモデルをざっくり試す----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

names = ['LogisticRegression',
  'SVC',
  'LinearSVC',
  'KNeighbors',
  'DecisionTree',
  'RandomForest',
  'MLPClassifier']

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

#各モデルを使いテスト

import statistics
result_list = []
for i in range(0,10):
    print('test {0}'.format(i))
    name, res = evaluate_models(df_forAgePred, l_pred, l_models)
    for x, y in zip(name, res):
        result_list.append([x, y])

#結果表示
print('mean squared error results by model')
print()
for n in names:
    r = [i[1] for i in result_list if i[0] == n]
    print(n)
    print('avg: {0:,.2f} / median: {1:,.2f}'.format(sum(r)/len(r), statistics.median(r)))


#Tensorflow 使用例--------------------------------------------------------------------------

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

for _ in range(1000):
  sess.run(train, {x: x_dataset, y: y_dataset})

# evaluate training accuracy
ans_W, ans_b, ans_loss = sess.run([W, b, loss], {x: x_dataset, y: y_dataset})
print("W: %s b: %s loss: %s"%(ans_W, ans_b, ans_loss))


# Tensorflow 使用例 2 ------------------------------------------------------------------
import sys
import tensorflow as tf

sys.stderr.write("*** start ***\n")

dataset_x = [[1.],[5.]]
dataset_y = [[4.],[2.]]

x = tf.placeholder("float", [None, 1])
y_ = tf.placeholder("float", [None, 1])

W = tf.Variable([1.])
b = tf.Variable([0.])
y = W*x + b

sess = tf.Session()
init = tf.global_variables_initializer()
loss = tf.reduce_sum(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

sess.run(init)
print('Initial state')
print('loss: ' + str(sess.run(loss, feed_dict={x: dataset_x, y_: dataset_y})))
print("W: %f, b: %f" % (sess.run( W ), sess.run( b )) )

for step in range(100):
    sess.run(train_step, feed_dict={x: dataset_x, y_: dataset_y})
    if (step+1) % 20 == 0:
        print('\nStep: %s' % (step+1))
        print('loss: ' + str(sess.run(loss, feed_dict={x: dataset_x, y_: dataset_y})))
        print("W: %f, b: %f" % (sess.run( W ), sess.run( b )) )



# Tensorflow 使用例 3 ------------------------------------------------------------------

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


# Tensorflow 使用例 4 ------------------------------------------------------------------

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


