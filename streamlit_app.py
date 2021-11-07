import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
#from sklearn import datasets
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


# digit_path = 'https://git.uwaterloo.ca/aaljmiai/ahdd1/-/raw/master/'
# y_test = pd.read_csv(digit_path + 'csvTestLabel_10k_x_1.csv')
# X_test = pd.read_csv(digit_path + 'csvTestImages_10k_x_784.csv')
#y_train = pd.read_csv(digit_path + 'csvTrainLabel_60k_x_1.csv')
#X_train = pd.read_csv(digit_path + 'csvTrainImages_60k_x_784.csv')

#st.write(cv2.__version__)
#
#st.success('Success message')
#
#st.write('---')
#st.write("""
## Arabic Handwritten Image Recognition
#
#""")
#st.write('---')
#
#
## image = Image.open('im.png')
## show = st.image(image, use_column_width=True)
#
#st.sidebar.title("Upload Image")
#
##Disabling warning
#st.set_option('deprecation.showfileUploaderEncoding', False)
#
#from ipysketch import Sketch
#Sketch('im')
#
##Choose your own image
#uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
#
#
#
##st.write(type( np.asarray(Image.open(uploaded_file))))
#
##im =  np.asarray(Image.open(uploaded_file))
#
vec_img = None

model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("gbm_n_estimators60000_objective_softmax_8_by_8_pix")

def ahmed(uploaded_file):
    if uploaded_file is not None:
        #st.image(uploaded_file, use_column_width=True)
  
        #u_img = Image.open(uploaded_file)
#          Image.show(u_img, 'Uploaded Image', use_column_width=True)
        # We preprocess the image to fit in algorithm.
        #st.write(type(uploaded_file))
        #img = np.asarray(u_img)
       
        # convert image to black and white pixels.
        #grayImage = 255 - cv2.cvtColor(uploaded_file, cv2.COLOR_BGR2GRAY)
        
        #plot the image to visualize the digit.
        #plt.imshow(grayImage)
        #plt.show()
        
        # flip the image up down to meet the image orientation of the training dataset.
        #grayImage = cv2.flip(grayImage,0)
#          grayImage = cv2.rotate(grayImage, cv2.cv.ROTATE_90_COUNTERCLOCKWISE)
        grayImage = np.flipud(np.rot90(uploaded_file,1))
        
        #plt.imshow(grayImage)
        #plt.show()
        #st.image(grayImage, use_column_width=True)
        
        # resize the orginal image to 28x28 as in the dataset
        # dsize
        width  = 8
        height = 8
        dsize = (width, height)
                # resize image
        output = cv2.resize(grayImage, dsize, interpolation = cv2.INTER_AREA)
        #plt.imshow(output)
        #plt.show()
        
        # vectorizing the image
        vec_img = output.reshape(1, -1)/255
        #st.image(output, use_column_width=True)
        
        #return model_xgb_2.predict(xgb.DMatrix(vec_img))
        return cv2.resize(vec_img.reshape(8,8), (224, 224), interpolation = cv2.INTER_AREA),  model_xgb_2.predict(xgb.DMatrix(vec_img))
  
  


#
#load_clf = pd.read_pickle('https://github.com/entaim/streamlit/raw/master/gbm_n_estimators60000_objective_softmax_8_by_8_pix.pickle')
#load_clf= load_model('gbm_n_estimators60000_objective_softmax_8_by_8_pix.pickle')


#prediction=model_xgb_2.predict(xgb.DMatrix(vec_img))
#st.write(prediction[0])
#st.write('---')


# Main Panel

# Print specified input parameters
#st.header('Specified Input parameters')
#st.write()
#st.write('---')

# st.set_option('deprecation.showPyplotGlobalUse', False)
# # Build Regression Model
# model = RandomForestRegressor(n_estimators=10, random_state=0)
# model.fit(X_train, y_train)
# # Apply Model to Make Prediction
# prediction = model.predict(df)

# st.header('Predicted Price (Saudi Riyal) :red_circle:')
# st.write(round(prediction[0], 2),"SR") 
# st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# st.write('* SHAPE values show how much a given feature changed our prediction')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')


# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')

# st.write('---')
# st.write('## Our Client :dizzy:')
# #st.write(x1)
# #st.write()
# st.write("""



# ![](https://user-images.githubusercontent.com/20365333/138615337-bfbdfdb2-494c-4265-8ff0-467b158f95d3.png)


# """)












import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
# from train import Net
from scipy.ndimage.interpolation import zoom
import os
import layout

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def main():
#     layout.footer()
    st.title("Handwritten Arabic Digit - Classification Model")
    left_column, right_column = st.columns(2)

    # st.write(model.eval())

    # Create a canvas component
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
        #grey = zoom(grey, 0.125)
#         st.image(grey, use_column_width=True)
        p = ahmed(grey)
        #x_np = torch.from_numpy(grey).unsqueeze(0) #
        #x = x_np.unsqueeze(0)
        #x = x.float()
        #output = model(x)
        #pred = torch.max(output, 1)
        #pred = pred[1].numpy()
    with right_column:
        st.header("Predicted Result")
        
        st.subheader('Pred# ')
        st.image(p[0], clamp=True)
    st.write(np.round(p[1][0], 3))

if __name__ == '__main__':
    main()
    import warnings
    warnings.filterwarnings('ignore')

