import math
import numpy as np
from PIL import Image
import cv2

def crop_face_from_scene(image, box, scale=1.3):
    y1,x1,w,h=box
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=np.int16(max(math.floor(y1),0))
    x1=np.int16(max(math.floor(x1),0))
    y2=np.int16(min(math.floor(y2),w_img))
    x2=np.int16(min(math.floor(x2),h_img))
    region=image[x1:x2,y1:y2]
    return Image.fromarray(region[:,:,::-1])

def crop_face(app, img):
    try:
        faces = app.get(img)
        box = faces[0][0], faces[0][1], faces[0][2]-faces[0][0], faces[0][3]-faces[0][1]
        if box[2] < 30 or box[3] < 30:
            return [False, 0]
        return [True, crop_face_from_scene(image=img, box=box)]
    except:
        return [False, 0]