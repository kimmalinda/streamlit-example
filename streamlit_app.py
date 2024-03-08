import streamlit as st
import joblib
#Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#Model
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def cExamPre():
    match a:
      case "ทบทวน อ่านหนังสือคนเดียว":
          b = "w_friend"
      case "ติวหนังสือกับกลุ่มเพื่อน":
          b = "own"
      case "ไม่อ่าน":
          b = "No"
    return b


def oh(df):
  df.ExamPrepare = df.ExamPrepare.apply(cExamPre)
  ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
  ohetransform = ohe.fit_transform(df[['gender','ExamPrepare']])
  df = pd.concat([df, ohetransform],axis =1)
  return df

#แทนค่าBehavior
def Behavior(c):
  match c:
    case "ไม่ขาดเรียนเลย":
      b = 3
    case "ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)":
      b = 2
    case "ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)":
      b = 1
    case "ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)":
      b = 0
  return b

def choice(d):
  match d:
    case "ชอบ" | "ทำ":
      b = 1
    case "ไม่ชอบ" | "ไม่ได้ทำ":
      b = 0
  return b


#เคลียร์GPA
def clearGPA(df):
  for i in df.index:
    if len(df['ClassGPAX'][i]) > 4:
      df['ClassGPAX'][i] = df['ClassGPAX'][i][-4:]
  return df

#แปลงGPAเป็นรูปแบบของคลาส
def c1GPAX(a):
  grade  = float(a)
  if 3.25 <= grade <= 4.00:
    b = 'A'
  elif 2.75 <= grade <= 3.24:
    b = 'B'
  elif 2.25 <= grade <= 2.74:
    b = 'C'
  else:
    b = 'D'
  return b

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


#แปลงเกรดเป็นเลข
def GradeToNum(e):
  match e:
    case "A":
      b = 4
    case "B+":
      b = 3.5
    case "B":
      b = 3
    case "C+":
      b = 2.5
    case "C":
      b = 2
    case "D+":
      b = 1.5
    case "D":
      b = 1
    case _:
      b = 0
  return b

def cmajor(u):
  match u:
    case "คณิตศาสตร์":
      b = "Math"
    case "สถิติ":
      b = "Stat"
    case "เคมี":
      b = "Chem"
    case "ชีววิทยา":
      b = "Bio"
    case "ฟิสิกส์":
      b = "Phys"
    case "ฟิสิกส์ประยุกต์":
      b = "App-Phys"
    case "วิทยาการคอมพิวเตอร์":
      b = "Com-Sci"
    case "เทคโนโลยีสารสนเทศ":
      b = "IT"
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

  match major:
    case "Math":
      mc = 252
    case "Stat":
      mc = 255
    case "Chem":
      mc = 256
    case "Bio":
      mc = 258
    case "Phys":
      mc = 261
    case "App-Phys":
      mc = 262
    case "Com-Sci":
      mc = 254
    case "IT":
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

df = oh(df)
df.major = df.major.apply(cmajor)
df.Good_math = df.Good_math.apply(choice)
df.part_time= df.part_time.apply(choice)
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
df.rename(columns = {'gender_ชาย':'male','gender_หญิง':'female'}, inplace = True)

result = SplitData(df,0.2)
X_train = result[0]
X_test = result[1]
y_train = result[2]
y_test = result[3]

svm = SVC(C=1,kernel='poly',gamma=0.1,degree=3)
svm.fit(X_train,y_train)
pred = svm.predict(X_test)

accuracy = svm.score(X_test,y_test))

joblib.dump(svm, 'svm_model.sav')

#On Web
st.title('การทำนายระดับผลการเรียน')
st.markdown('กรุณากรอกข้อมูลให้ครบเพื่อใช้ในการทำนาย')
st.image('img.png',caption='ขั้นตอนการกรอกผลการเรียนแต่ละวิชาชั้นปีที่ 1')
st.link_button("Go to Reg","https://reg9.nu.ac.th/registrar/home.asp")
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
  MajorBe1 = st.selectbox("พฤติกรรมการเข้าเรียนในรายวิชาบังคับ(วิชาเอก)",('ไม่ขาดเรียนเลย','ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)','ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)','ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)'))
  OtherBe1 = st.selectbox("พฤติกรรมการเข้าเรียนในรายวิชาอื่นๆ ",('ไม่ขาดเรียนเลย','ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)','ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)','ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)'))
  ExamPrepare1 = st.selectbox("เตรียมตัวสอบอย่างไร",('ทบทวน อ่านหนังสือคนเดียว','ติวหนังสือกับกลุ่มเพื่อน','ไม่อ่าน'))

re = st.button('Predict class of grade')


def predict(data):
  svm = joblib.load('svm_model.sav')
  return svm.predict(data)

if r:
  result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
  st.text(result[0])




