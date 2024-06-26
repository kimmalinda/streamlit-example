import streamlit as st
import joblib
#Data
import numpy as np
import pandas as pd
import math
import random
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#On Web
st.title('การพยากรณ์ระดับผลการเรียนเฉลี่ยตลอดหลักสูตรจากผลการเรียนแต่ละรายวิชาในชั้นปีที่ 1')
st.markdown('กรุณากรอกข้อมูลให้ครบถ้วนเพื่อใช้ในการทำนาย')
st.image("ขั้นตอน.png",caption='ขั้นตอนวิธีการกรอกผลการเรียนแต่รายวิชาในชั้นปีที่ 1')
col1, col2 = st.columns(2)
with col1:
  major1 = st.selectbox("สาขาที่เรียน",('คณิตศาสตร์','สถิติ','ฟิสิกส์','ฟิสิกส์ประยุกต์','เคมี','ชีววิทยา','วิทยาการคอมพิวเตอร์','เทคโนโลยีสารสนเทศ'))
with col2:
  grade1 = st.text_input('ผลการเรียนแต่รายวิชาในชั้นปีที่ 1')
  
re = st.button('Predict class of grade')


#predict
def predict(data):
  model_svm = joblib.load('svm_model.sav')
  return model_svm.predict(data)
  
def cGPAX(a):
  grade  = float(a)
  if 3.25 <= grade <= 4.00:
    b = 'Honor Class'
  elif 2.75 <= grade <= 3.24:
    b = 'Medium Class'
  elif 2.25 <= grade <= 2.74:
    b = 'Lower Class'
  else:
    b = 'Beware Class'
  return b

def GradeToNum(a):
  if a == "A":
    b = 4
  elif a =="B+":
    b = 3.5
  elif a =="B":
    b = 3
  elif a =="C+":
    b = 2.5
  elif a =="C":
    b = 2
  elif a =="D+":
    b = 1.5
  elif a == "D":
    b = 1
  else:
    b = 0
  return b

def CleanText(text0,major):
  course=[]
  credit=[]
  grade=[]
  text = text0.replace("\t", " ")
  text = text.replace("\n"," ")
  text = text.replace("/"," ")
  text = text.split(" ")
  text =list(filter(None, text))
  num_of_row = len(text)
  for i in range(num_of_row):
    if text[i][0].isnumeric() and len(text[i]) == 6:
      course.append(int(text[i]))
      for j in range(i+1,i+10):
        if text[j].isnumeric() and not text[j+1].isnumeric():
          credit.append(int(text[j]))
          grade.append(text[j+1])
          break

  df1 = pd.DataFrame()
  df1 = pd.DataFrame(columns=['Course', 'Credit','Grade'])
  df1['Course'] = course
  df1['Credit'] = credit
  df1['Grade'] = grade
  df1.Grade = df1.Grade.apply(GradeToNum)


  if major == 'คณิตศาสตร์':
    mc = 252
  elif major == "สถิติ":
    mc = 255
  elif major ==  "เคมี":
    mc = 256
  elif major ==  "ชีววิทยา":
    mc = 258
  elif major ==  "ฟิสิกส์":
    mc = 261
  elif major ==  "ฟิสิกส์ประยุกต์":
    mc = 262
  elif major == "วิทยาการคอมพิวเตอร์":
    mc = 254
  elif major ==  "เทคโนโลยีสารสนเทศ":
    mc = 273
    
  genEdgrade = 0
  genEdcredit = 0
  majorgrade = 0
  majorcredit = 0
  othergrade = 0
  othercredit = 0


  for i in range(len(df1)):
    if df1['Course'][i] // 1000 == 1:
      if df1['Course'][i] != 1281:
        A = genEdcredit * genEdgrade
        genEdcredit = genEdcredit + df1['Credit'][i]
        genEdgrade = (A + df1['Credit'][i]*df1['Grade'][i])/genEdcredit
      else:
          continue
    elif df1['Course'][i] // 1000 == mc:
      B = majorcredit * majorgrade
      majorcredit = majorcredit + df1['Credit'][i]
      majorgrade = (B + df1['Credit'][i]*df1['Grade'][i])/majorcredit
    else:
      C = othercredit * othergrade
      othercredit = othercredit + df1['Credit'][i]
      othergrade = (C + df1['Credit'][i]*df1['Grade'][i])/othercredit
  return [genEdgrade,majorgrade,othergrade]

def SplitData(df,test_size):
  total_index = list(df.index)
  A =[]
  B =[]
  C=[]
  D=[]
  for i in total_index:
    if df['ClassGPAX'][i] == 'Honor Class':
      A.append(i)
    elif df['ClassGPAX'][i] == 'Medium Class':
      B.append(i)
    elif df['ClassGPAX'][i] == 'Lower Class':
      C.append(i)
    else:
      D.append(i)

  nA = math.floor(test_size*(len(A)))
  nB = math.floor(test_size*(len(B)))
  nC = math.floor(test_size*(len(C)))
  nD = math.floor(test_size*(len(D)))
  random.shuffle(A)
  random.shuffle(B)
  random.shuffle(C)
  random.shuffle(D)
  test_index = A[:nA] + B[:nB] + C[:nC] + D[:nD]
  train_index = list(set(total_index)-set(test_index))

  #result = []
  df_test = df.loc[test_index]
  df_train = df.loc[train_index]
  X_train = df_train.drop({'ClassGPAX'},axis = 1)
  X_test = df_test.drop({'ClassGPAX'},axis = 1)
  y_train = df_train['ClassGPAX']
  y_test = df_test['ClassGPAX']

  return [X_train,X_test,y_train,y_test]

df = pd.read_csv(r'ver412.csv')
#เพิ่มคอลัมเกรด
df['GPAGenEd'] = None
df['GPAMajor'] = None
df['GPAOther'] = None

df.ClassGPAX = df.ClassGPAX.apply(cGPAX)
for ind in df.index:
  res = CleanText(df['gradeText'][ind],df['major'][ind])
  df['GPAGenEd'][ind] = res[0]
  df['GPAMajor'][ind] = res[1]
  df['GPAOther'][ind] = res[2]

df = df.drop(df.columns[[0,1]], axis=1)



result = SplitData(df,0.2)
X_train = result[0]
X_test = result[1]
y_train = result[2]
y_test = result[3]


model_svm = SVC(C=1,kernel='poly',gamma=0.1,degree=3)
model_svm.fit(X_train,y_train)
pred = model_svm.predict(X_test)

acc = model_svm.score(X_test,y_test)

joblib.dump(model_svm, 'svm_model.sav')

#เตรียมข้อมูลเบื้องต้น
df1 = pd.DataFrame()
df1 = pd.DataFrame(columns=['grade','major','gradeGenEd','gradeMajor','gradeOther'])
grade=[]
major=[]
grade.append(grade1)
major.append(major1)
df1['grade'] = grade
df1['major'] = major
df1['gradeGenEd'] = None
df1['gradeMajor'] = None
df1['gradeOther'] = None

#คลีนข้อมูลขั้นต้น
for ind in df1.index:
  res = CleanText(df1['grade'][ind],df1['major'][ind])
  df1['gradeGenEd'][ind] = res[0]
  df1['gradeMajor'][ind] = res[1]
  df1['gradeOther'][ind] = res[2]

df1= df1.drop(df1.columns[[0,1]], axis=1)
arr = np.array([[df1.gradeGenEd[0],df1.gradeMajor[0],df1.gradeOther[0]]])

#result
if re:
  if len(grade1) < 100:
    st.text('กรุณากรอกผลการเรียนให้ครบถ้วน')
  else:
    result = predict(arr)
    st.text([result[0]])
    st.write("เกณฑ์การให้ระดับผลการเรียนเฉลี่ยตลอดหลักสูตรดังนี้")
    col1,mid, col2 = st.columns([1,1,50])
    with col1:
      st.image('มง.png', width=20)
      st.image('m.png', width=20)
      st.image('Lo.png', width=20)
      st.image('be.png', width=20)
    with col2:
      st.text('Honor Class หมายถึง ผลการเรียนเฉลี่ยตั้งแต่ 3.25 ขึ้นไป')
      st.text('Medium Class หมายถึง ผลการเรียนเฉลี่ยตั้งแต่ 2.75 - 3.24')
      st.text('Lower Class หมายถึง ผลการเรียนเฉลี่ยตั้งแต่ 2.25 - 2.74')
      st.text('Beware Class หมายถึง ผลการเรียนเฉลี่ยต่ำกว่า 2.25')

