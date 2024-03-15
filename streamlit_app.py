import streamlit as st
import joblib
#Data
import numpy as np
import pandas as pd
import math
import random
from sklearn.preprocessing import OneHotEncoder
#Model
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

st.title('การทำนายระดับผลการเรียน')
st.markdown('กรุณากรอกข้อมูลให้ครบเพื่อใช้ในการทำนาย')
st.header('ผลการเรียนแต่ละรายวิชาชั้นปีที่ 1')
col1, col2 = st.columns(2)
with col1:
  st.text('ผลการเรียนชั้นปีที่ 1')
  grade1 = st.text_input('')
with col2:
  st.text('พฤติกรรมที่ส่งผลต่อผลการเรียน')
  gender1 = st.selectbox("เพศ",('ชาย','หญิง'))
  part_time1 = st.selectbox("part_time",('ทำ','ไม่ทำ'))
  fav1 = st.selectbox("ชอบสาขาที่เรียนหรือไม่",('ชอบ','ไม่ชอบ'))
  GenEdBe1 = st.selectbox("พฤติกรรมการเข้าเรียนในรายวิชาศึกษาทั่วไป(วิชามอ)",('ไม่ขาดเรียนเลย','ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)','ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)','ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)'))
  MajorBe1 = st.selectbox("พฤติกรรมการเข้าเรียนในรายวิชาเอก",('ไม่ขาดเรียนเลย','ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)','ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)','ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)'))
  OtherBe1 = st.selectbox("พฤติกรรมการเข้าเรียนในรายวิชาอื่น ๆ ",('ไม่ขาดเรียนเลย','ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)','ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)','ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)'))
  ExamPrepare1 = st.selectbox("เตรียมตัวสอบอย่างไร",('ทบทวน อ่านหนังสือคนเดียว','ติวหนังสือกับกลุ่มเพื่อน','ไม่อ่าน'))

re = st.button('Predict class of grade')

#predict
def predict(data):
  model_svm = joblib.load('svm_model.sav')
  return model_svm.predict(data)

def Behavior(a):
  if a == "ไม่ขาดเรียนเลย":
    b = 3
  elif a =="ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)":
    b = 2
  elif a == "ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)":
    b = 1
  else: 
    b = 0
  return b

def cExamPre(a):
  if a =="ทบทวน อ่านหนังสือคนเดียว":
    b = "w_friend"
  elif a== "ติวหนังสือกับกลุ่มเพื่อน":
    b = "own"
  else: 
    b = "No"
  return b

def oh(df):
  df.ExamPrepare = df.ExamPrepare.apply(cExamPre)
  ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
  ohetransform = ohe.fit_transform(df[['gender','ExamPrepare']])
  df = pd.concat([df, ohetransform],axis =1)
  return df

def choice1(a):
  if a == "ทำ":
    b = 1
  elif a =="ไม่ได้ทำ":
    b = 0
  return b

def choice2(a):
  if a == "ชอบ":
    b = 1
  elif a=="ไม่ชอบ":
    b = 0
  return b

def clearGPA(df):
  for i in df.index:
    if len(df['ClassGPAX'][i]) > 4:
      df['ClassGPAX'][i] = df['ClassGPAX'][i][-4:]
  return df

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
  GradeToNum()
  
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
    if df['ClassGPAX'][i] == 'A':
      A.append(i)
    elif df['ClassGPAX'][i] == 'B':
      B.append(i)
    elif df['ClassGPAX'][i] == 'C':
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

df = pd.read_csv(r'ver57math_BE.csv')

df['GPAGenEd'] = None
df['GPAMajor'] = None
df['GPAOther'] = None

df = oh(df)
df.part_time= df.part_time.apply(choice1)
df.Good_math = df.Good_math.apply(choice2)
df.GenEdCA = df.GenEdCA.apply(Behavior)
df.MajorCA = df.MajorCA.apply(Behavior)
df.OtherCA = df.OtherCA.apply(Behavior)
clearGPA(df)
df.ClassGPAX = df.ClassGPAX.apply(cGPAX)
for ind in df.index:
  res = CleanText(df['gradeText'][ind],df['major'][ind])
  df['GPAGenEd'][ind] = res[0]
  df['GPAMajor'][ind] = res[1]
  df['GPAOther'][ind] = res[2]
df = df.drop(df.columns[[0,1,3,9]], axis=1)

st.text(df)


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

#แทนค่าด้วยเลขOneHotEncoder
p7=0
p8=0
p9=0
p10=0
p11=0
p13=0
p14=0
p15=0

if gender1 == 'ชาย':
  p7 = 1
  p8 = 0
else:
  p7 = 0
  p8 = 1
if part_time1  == 'ทำ':
  p9 = 1
  p10 = 0
else:
  p9 = 0
  p10 = 1
if fav1 == 'ชอบ':
  p11 = 1
  p12 = 0
else:
  p11 = 0
  p12 = 1
if ExamPrepare1 == 'ติวหนังสือกับกลุ่มเพื่อน':
  p13 = 1
  p14 = 0
  p15 = 0
elif ExamPrepare1 == 'ทบทวน อ่านหนังสือคนเดียว':
  p13 = 0
  p14 = 1
  p15 = 0
else:
  p13 = 0
  p14 = 0
  p15 = 1

df = pd.DataFrame()
df = pd.DataFrame(columns=['grade','GenEdBe','MajorBe','OtherBe','gradeGenEd','gradeMajor','gradeOther'])
grade=[]
GenEdBe=[]
MajorBe=[]
OtherBe=[]
grade.append(grade1)
GenEdBe.append(GenEdBe1)
MajorBe.append(MajorBe1)
OtherBe.append(OtherBe1)
df['grade'] = grade
df['GenEdBe'] = GenEdBe
df['MajorBe'] = MajorBe
df['OtherBe'] = OtherBe
df['gradeGenEd'] = None
df['gradeMajor'] = None
df['gradeOther'] = None

#คลีนข้อมูล57
Behavior(df.GenEdBe) 
Behavior(df.MajorBe) 
Behavior(df.OtherBe) 

for ind in df.index:
  res = CleanText(df['gradeText'][ind],df['major'][ind])
  df['gradeGenEd'][ind] = res[0]
  df['gradeMajor'][ind] = res[1]
  df['gradeOther'][ind] = res[2]

df_pred = df.drop(df.columns[[0]], axis=1)
arr = np.array([[df_pred.GenEdBe[0],df_pred.MajorBe[0],df_pred.OtherBe[0],df_pred.gradeGenEd[0],df_pred.gradeMajor[0],df_pred.gradeOther[0],
p7,p8,p9,p10,p11,p12,p13,p14,p15]])

#result
if re:
  result = predict(arr)
  st.text(Class[result[0]])
