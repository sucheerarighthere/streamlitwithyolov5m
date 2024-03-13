import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect


#ตั้งค่าเพจให้เป็นแบบที่เราต้องการ พื้นหกลัง ตัวหนังสือ ใดๆว่าไป
st.set_page_config(page_title="Object Detection",  # Setting page title
    page_icon="🔬",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",# Expanding sidebar by default
    
        )   

#ตั้งค่าภาพ
image = Image.open('STAT-Header-Logo-V7.png')
st.image(image, caption='สาขาวิชาสถิติ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น', use_column_width=True )

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestyolo.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
  
  file_bytes = np.asarray(bytearray(uploaded_file.read()))
  image = cv2.imdecode(file_bytes, 1)

  imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  #st.image(imgRGB)

  st.write("")
  st.write("Detecting...")
  result = model(imgRGB, size=300)
  
  detect_class = result.pandas().xyxy[0] 


  # #labels, cord_thres = detect_class[:, :].numpy(), detect_class[:, :].numpy()
  
  # #     xmin       ymin    xmax        ymax          confidence  class    name
  # #0  148.605362   0.0    1022.523743  818.618286    0.813045      2      turtle
  
  #  st.code(detect_class[['name', 'xmin','ymin', 'xmax', 'ymax']])
  
  
  
  # st.success(detect_class)
  
  outputpath = 'output.jpg'
  num_objects_detected = len(detect_class)
  result.render()  # render bbox in image
  for im in result.ims:
      im_base64 = Image.fromarray(im)
      im_base64.save(outputpath)
      img_ = Image.open(outputpath)
      st.image(img_, caption='Model Prediction(s)')
      st.write(f"Number of objects detected: {num_objects_detected}")
