import onnxruntime as ort
import streamlit as st
import numpy as np
from PIL import Image

from retina_face import *
from tranformers import *


# retina_face detecter
face_detecter = FaceAnalysis()
face_detecter.prepare()

# FAS model
model_path = './weights/best_model.onnx'
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_names = [i.name for i in sess.get_inputs()]
input_shape = sess.get_inputs()[0].shape

def inference(img):
    image = crop_face(face_detecter, img)[1]
    transformed_image = transform(image)
    input_tensor = transformed_image.reshape(input_shape).cpu().numpy()
    outputs = sess.run(None, {input_names[0]: input_tensor, input_names[1]: input_tensor})
    outputs = softmax(outputs[0][0])
    if outputs[0]>=outputs[1]:
        # print('Spoof\n', 'Confident: {}%'.format(outputs[0]))
        return([image, 'giả mạo :thumbsdown:', str(outputs[0])+'%'])
    else:
        # print('Live\n', 'Confident: {}%'.format(outputs[1]))
        return([image, 'thật :thumbsup:', str(outputs[1])+'%'])
    

def handle(img):
    image = Image.open(img).convert('RGB')
    image_res = image.resize((400, int(image.size[1]/image.size[0]*400)))
    image = np.array(image)[:,:,::-1]
    image_res = np.array(image_res)
    col1.write("#### Ảnh gốc :camera:")
    col1.image(image_res, channels='RGB')

    if crop_face(face_detecter, image)[0] == False:
        st.error("File tải lên không chứa hình ảnh khuôn mặt người. Vui lòng thử lại với ảnh khác.")
    else:
        img_crop, res, conf = inference(image)
        img_crop_res = img_crop.resize((350, int(img_crop.size[1]/img_crop.size[0]*350)))
        col2.write("#### Kết quả nhận diện :clipboard:")
        col2.write("##### Phần ảnh chứa khuôn mặt người :scissors:")
        col2.image(img_crop_res, channels='RGB')
        col2.write('##### Nhãn: '+res)
        col2.write('##### Độ tin cậy: '+conf)

st.set_page_config(layout="wide", page_title="Hệ thống chống giả mạo khuôn mặt")

st.write("## Nhận diện tính xác thực của ảnh màu có chứa khuôn mặt người")
st.write("Phiên bản thử nghiệm")
st.sidebar.write("## Tải ảnh có chứa khuôn mặt người :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

col1, col2 = st.columns(2, gap='large')
my_upload = st.sidebar.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("Tệp tải lên quá lớn. Vui lòng tải lên tệp có dung lượng nhỏ hơn 5MB.")
    else:
        handle(img=my_upload)
