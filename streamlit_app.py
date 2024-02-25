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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv(r'ver.sci412.csv')
#เพิ่มคอลัมเกรด
df['gradeGenEd'] = None
df['gradeMajor'] = None
df['gradeOther'] = None
#แทนค่าด้วยเลขOneHotEncoder
def replace(df):
  ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
  ohetransform = ohe.fit_transform(df[['gender','part-time','fav','ExamPrepare']])
  df = pd.concat([df, ohetransform],axis =1).drop(df.columns[[1,2,5,9]], axis=1)
  return df

#แทนค่าBehavior
def Behavior(a):
    match a:
      case "ไม่ขาดเรียนเลย":
        b = 3
      case "ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)":
        b = 2
      case "ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)":
        b = 1
      case "ขาดเรียนเป็นส่วนใหญ่ (ขาดเกินครึ่งของภาคเรียน)":
        b = 0
    return b

#เคลียร์GPA
def clearGPA(DF):
  for i in DF.index:
    if len(DF['GPA'][i]) > 4:
      DF['GPA'][i] = DF['GPA'][i][-4:]
  return DF

#แปลงGPAเป็นรูปแบบของคลาส
def exGPA(a):
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

DF=df
DF.GPA = DF.GPA.apply(exGPA)

for ind in DF.index:
  CleanText(DF['grade'][ind])
  res = GradeGroup(CleanText(DF['grade'][ind]),DF['major'][ind])
  DF['gradeGenEd'][ind] = res[0]
  DF['gradeMajor'][ind] = res[1]
  DF['gradeOther'][ind] = res[2]

df_pred = DF.drop(DF.columns[[0,1]], axis=1)

X = df_pred.drop({'GPA'},axis = 1)
y = df_pred['GPA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


model_nb = GaussianNB(0.050068723840508415)
model_nb.fit(X_train,y_train)
pred = model_nb.predict(X_test)

print(model_nb.score(X_test,y_test))
print(classification_report(y_test,pred))
print(confusion_matrix(y_test, pred))

# print best parameter after tuning
print(model_nb.best_params_)

# print how our model looks after hyper-parameter tuning
print(model_nb.best_estimator_)

import joblib
joblib.dump(model_rf, 'rf_model.sav')

