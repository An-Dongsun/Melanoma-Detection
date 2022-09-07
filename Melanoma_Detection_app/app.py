import flask
from flask import Flask, request, render_template

# 필요할 라이브러리 불러오기
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


app = Flask(__name__)
   
# index 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')
   
# 이미지 업로드에 대한 예측값 반환
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")
        
        img_array = keras.preprocessing.image.img_to_array(file)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        model = load_model('Melanoma_Detection_app/EfficientNet_V2B1_model5.h5')

        predictions = model.predict(img_array)
        score = predictions[0]

        class_names = ('nevus(모반)', 'malignant(흑색종 악성)', 'benign(흑색종 양성)', 'seborrheic_keratosis(지루성 각화증)', 'BCC(기저 세포 암종)')

        for i in range(len(class_names)):
            if np.argmax(score) == i:
                # 결과 리턴
                return render_template('index.html', label=class_names[i])
    
            
   
# 미리 학습시켜서 만들어둔 모델 로드
if __name__ == '__main__':
    model = load_model('Melanoma_Detection_app/EfficientNet_V2B1_model5.h5')
    app.run(host='0.0.0.0', port=8000, debug=True)