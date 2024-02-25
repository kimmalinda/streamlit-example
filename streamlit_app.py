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

