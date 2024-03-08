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
from sklearn.ensemble import RandomForestClassifier

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

def Behavior(a):
  if ExamPrepare == "ไม่ขาดเรียนเลย":
    b = 3
  elif ExamPrepare =="ขาดเรียนบ้างเล็กน้อย (ขาดเรียนไม่เกิน 3 ครั้งของภาคเรียน)":
    b = 2
  elif ExamPrepare == "ขาดเรียนระดับปานกลาง (ขาดเรียนเกิน 3 ครั้ง แต่ไม่ถึงครึ่งของภาคเรียน)":
    b = 1
  else: 
    b = 0
  return b

def choice(a):
  if "ชอบ" | "ทำ":
    b = 1
  elif "ไม่ชอบ" | "ไม่ได้ทำ":
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
