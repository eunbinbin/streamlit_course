import streamlit as st
import os
import openai
from PIL import Image, ImageTk
import requests, io


def generate():
    openai.api_key = "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"
    user_prompt = st.session_state['prompt']
    user_prompt += "in style: " + st.session_state['img_style']

    response = openai.Image.create(
        prompt=user_prompt,
        n=st.session_state['img_num'],
        size=st.session_state['img_size']
    )

    image_urls = []
    for i in range(len(response['data'])):
        image_urls.append(response['data'][i]['url'])
        print(image_urls)

    i = 1
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        st.text(str(i) + '번째 이미지')
        st.image(image=image)
        i += 1


col1, col2 = st.columns(2)

with col1:
    st.title('AI 이미지 생성기')
    st.text_input('프롬프트', key='prompt')
    st.number_input('이미지 개수', value=1, key='img_num')
    st.selectbox('이미지 크기', options=['256x256', '512x512', '1024x1024'], key='img_size')
    st.radio('이미지 스타일', options=['Realistic', 'Cartoon', '3D Illustration', 'Flat Art'], key='img_style')
    btn = st.button('생성')

with col2:
    if btn:
        st.subheader('이미지 생성 결과')
        generate()