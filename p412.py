import streamlit as st
import joblib
#Data
import numpy as np
import pandas as pd


#On Web
st.title('การทำนายระดับผลการเรียน')
st.markdown('กรุณากรอกข้อมูลให้ครบเพื่อใช้ในการทำนาย')
st.header('ผลการเรียนแต่ละรายวิชาชั้นปีที่ 1 ')
col1, col2 = st.columns(2)
with col1:
  st.text('สาขา')
  major1 = st.selectbox("สาขาที่เรียน",('คณิตศาสตร์','สถิติ','ฟิสิกส์','ฟิสิกส์ประยุกต์','เคมี','ชีววิทยา','วิทยาการคอมพิวเตอร์','เทคโนโลยีสารสนเทศ'))
with col2:
  st.text('เกรดแต่ละรายวิชาชั้นปีที่ 1')
  grade1 = st.text_input('copy ผลการเรียนชั้นปีที่ 1')
  
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
  cExamPre(df.ExamPrepare)
  ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
  ohetransform = ohe.fit_transform(df[['gender','ExamPrepare']])
  df = pd.concat([df, ohetransform],axis =1)
  return df

def gender():
  if df.gender == 'ชาย':
    b = 0
  elif df.gender == 'หญิง':
    b=1
  return b

def choice(a):
  if df['part_time','Good_math'] == "ชอบ" | "ทำ":
    b = 1
  elif df['part_time','Good_math'] =="ไม่ชอบ" | "ไม่ได้ทำ":
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

def GradeToNum():
  if df1.Grade == "A":
    b = 4
  elif df1.Grade =="B+":
    b = 3.5
  elif df1.Grade =="B":
    b = 3
  elif df1.Grade =="C+":
    b = 2.5
  elif df1.Grade =="C":
    b = 2
  elif df1.Grade =="D+":
    b = 1.5
  elif df1.Grade == "D":
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
#เพิ่มคอลัมเกรด
df['GPAGenEd'] = None
df['GPAMajor'] = None
df['GPAOther'] = None


choice(df)
Behavior(df.GenEdCA)
Behavior(df.MajorCA)
Behavior(df.OtherCA)
clearGPA(df)
cGPAX(df.ClassGPAX) 
for ind in df.index:
  res = CleanText(df['gradeText'][ind],df['major'][ind])
  df['GPAGenEd'][ind] = res[0]
  df['GPAMajor'][ind] = res[1]
  df['GPAOther'][ind] = res[2]
  
df = df.drop(df.columns[[0,1,3,9]], axis=1)
df.rename(columns = {'gender_ชาย':'male','gender_หญิง':'female'}, inplace = True)


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
