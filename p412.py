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
