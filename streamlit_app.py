import streamlit as st
import pandas as pd
import numpy as np
import joblib

#On Web
st.title('การทำนายระดับผลการเรียน')
st.markdown('กรุณากรอกข้อมูลให้ครบเพื่อใช้ในการทำนาย')
st.image('img.png',caption='ขั้นตอนการกรอกผลการเรียนแต่ละวิชาชั้นปีที่ 1')
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
  model_rf = joblib.load('rf_model.sav')
  return model_rf.predict(data)

#แปลงเกรดเป็นเลข
def regrade(a):
  match a:
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

#รหัสตามสาขา
def majorcode(a):
  match a:
    case "คณิตศาสตร์":
      b = 252
    case "สถิติ":
      b = 255
    case "เคมี":
      b = 256
    case "ชีววิทยา":
      b = 258
    case "ฟิสิกส์":
      b = 261
    case "ฟิสิกส์ประยุกต์":
      b = 262
    case "วิทยาการคอมพิวเตอร์":
      b = 254
    case "เทคโนโลยีสารสนเทศ":
      b = 273
  return b

#พาร์ทคลีนข้อความ
def CleanText(text0):
  course=[]
  credit=[]
  grade=[]
  text = text0.replace("\t", " ")
  text = text.replace("\n"," ")
  text = text.replace("/"," ")
  text = text.split(" ")
  text =list(filter(None, text))
  for i in range(len(text)):
    if text[i][0].isnumeric() and len(text[i]) == 6:
      course.append(text[i])
      for j in range(i+1,i+10):
        if text[j].isnumeric():
          credit.append(text[j])
          grade.append(text[j+1])
          break

  df1 = pd.DataFrame()
  df1 = pd.DataFrame(columns=['Course', 'Credit','Grade'])
  df1['Course'] = course
  df1['Credit'] = credit
  df1['Grade'] = grade
  df1.Grade = df1.Grade.apply(regrade)
  return df1

def GradeGroup(df1,major):

  mj = majorcode(major)

  df1['Course'] = df1['Course'].astype(int)
  df1['Credit'] = df1['Credit'].astype(int)
  genEdgrade1 = 0
  genEdcredit1 = 0
  genEdgrade2 = 0
  genEdcredit2 = 0
  genEdgrade3 = 0
  genEdcredit3 = 0
  i = 0

  num_of_rows = len(df1)

  for i in range(num_of_rows):
    if df1['Course'][i] // 1000 == 1:
      if df1['Course'][i] != 1281:
        A = genEdcredit1 * genEdgrade1
        genEdcredit1 = genEdcredit1 + df1['Credit'][i]
        genEdgrade1 = (A + df1['Credit'][i]*df1['Grade'][i])/genEdcredit1
        #df['GenEd'][ind] = genEdgrade1
        #return genEdgrade1
      else:
          continue
    elif df1['Course'][i] // 1000 == mj:
      B = genEdcredit2 * genEdgrade2
      genEdcredit2 = genEdcredit2 + df1['Credit'][i]
      genEdgrade2 = (B + df1['Credit'][i]*df1['Grade'][i])/genEdcredit2
      #df['Major'][ind] = genEdgrade2
    else:
      C = genEdcredit3 * genEdgrade3
      genEdcredit3 = genEdcredit3 + df1['Credit'][i]
      genEdgrade3 = (C + df1['Credit'][i]*df1['Grade'][i])/genEdcredit3
      #df['NonEd'][ind] = genEdgrade3
  return [genEdgrade1,genEdgrade2,genEdgrade3]

#dictClass
Class = {'A':'Honor Class','B':'Medium Class','C':'Lower Class','D':'Beware'}

#เตรียมข้อมูลเบื้องต้น
df = pd.DataFrame()
df = pd.DataFrame(columns=['grade','major','gradeGenEd','gradeMajor','gradeOther'])
grade=[]
major=[]
grade.append(grade1)
major.append(major1)
df['grade'] = grade
df['major'] = major
df['gradeGenEd'] = None
df['gradeMajor'] = None
df['gradeOther'] = None

#คลีนข้อมูลขั้นต้น
for ind in df.index:
  CleanText(df['grade'][ind])
  res = GradeGroup(CleanText(df['grade'][ind]),df['major'][ind])
  df['gradeGenEd'][ind] = res[0]
  df['gradeMajor'][ind] = res[1]
  df['gradeOther'][ind] = res[2]

df_pred = df.drop(df.columns[[0,1]], axis=1)
arr = np.array([[df_pred.gradeGenEd[0],df_pred.gradeMajor[0],df_pred.gradeOther[0]]])

#result
if re:
  result = predict(arr)
  st.text(Class[result[0]])


