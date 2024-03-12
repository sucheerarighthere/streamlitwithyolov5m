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

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg",accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()))
        image = cv2.imdecode(file_bytes, 1)

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("")
        st.write("Detecting...")
        result = model(imgRGB, size=300)

        detect_class = result.pandas().xyxy[0]

        outputpath = 'output.jpg'
        num_objects_detected = len(detect_class)
        result.render()  # render bbox in image
        for im in result.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)
            img_ = Image.open(outputpath)

            # ใช้ st.beta_columns เพื่อจัดวางภาพและข้อความ
            col1, col2, col3 = st.beta_columns(3)
            col1.image(img_, caption='Original Image', use_column_width=True)
            col2.image(img_, caption='Model Prediction(s)', use_column_width=True)
            col3.markdown(f"<p style='font-size:24px; font-weight:bold;'>Number of objects detected: {num_objects_detected}</p>", unsafe_allow_html=True)
