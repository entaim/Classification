import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
import pickle 
import xgboost as xgb
import cv2
from PIL import Image

import numpy as np
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage.interpolation import zoom
import os
import layout



vec_img = None

model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("gbm_n_estimators60000_objective_softmax_8_by_8_pix")

def ahmed(uploaded_file):
    if uploaded_file is not None:
        grayImage = np.flipud(np.rot90(uploaded_file,1))
        width  = 8
        height = 8
        dsize = (width, height)
                # resize image
        output = cv2.resize(grayImage, dsize, interpolation = cv2.INTER_AREA)
        
        # vectorizing the image
        vec_img = output.reshape(1, -1)/255
        return cv2.resize(vec_img.reshape(8,8), (224, 224), interpolation = cv2.INTER_AREA),  model_xgb_2.predict(xgb.DMatrix(vec_img))



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def main():
    st.title("Handwritten Arabic Digit - Classification Model")
    left_column, right_column = st.columns(2)

    with left_column:
        st.header("Draw a number")
        st.subheader("[0-9]")
        canvas_result = st_canvas(
                fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
                # stroke_width="1, 25, 3",
                stroke_width = 10,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                width=224,
                height=224,
                drawing_mode="freedraw",
                key="canvas",
        )
    p = None
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        grey = rgb2gray(img)
        p = ahmed(grey)
    with right_column:
        st.header("Predicted Result")
        
        st.subheader('Pred# ')
        st.image(p[0], clamp=True)
    st.write(np.round(p[1][0], 3))

if __name__ == '__main__':
    main()
    import warnings
    warnings.filterwarnings('ignore')

