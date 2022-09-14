import numpy as np
import streamlit as st
st.set_page_config(page_title='Face Compare', layout='wide')

from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

st.title('Welcome to Face Compare!')
st.write('Upload two images to compare.')

img_1 = st.file_uploader('Upload the first image')
img_2 = st.file_uploader('Upload the second image')

img_1 = Image.open(img_1)
img_2 = Image.open(img_2)

img_1_cropped = mtcnn(img_1)
img_2_cropped = mtcnn(img_2)

img_1_embedding = resnet(img_1_cropped.unsqueeze(0)).detach().numpy()[0]
img_2_embedding = resnet(img_2_cropped.unsqueeze(0)).detach().numpy()[0]

prob = np.dot(img_1_embedding, img_2_embedding)

st.write('Similarity Score: {}%'.format(str(prob*100)[:4]))