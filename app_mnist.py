import os.path
# pip install opencv-python
import cv2
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# pip install streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas

# 학습된 모델 불러오기
model_file = 'mnist_model.h5'
model = load_model(model_file)

# 헤더 출력
st.subheader('손글씨 숫자 인식')

# canvas 입력
SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode='freedraw',
    update_streamlit=False,
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('모델 입력 형태')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = model.predict(np.reshape(test_x, (1, 28, 28)))
    st.success(np.argmax(res[0]))
    st.bar_chart(res[0])




