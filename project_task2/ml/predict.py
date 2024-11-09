import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print("Model loaded successfully")
    return model

def prediction(img_path, model_path, indices_path):

    IMAGE_SIZE = 96   

    #loading the indices of labels
    indices = {}
    with open(indices_path, 'rb') as fp:
        indices = pickle.load(fp)


    #reading the image and normalizing it
    model = load_model(model_path=model_path)
    img = Image.open(img_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)/255.0

    #logits
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    
    #turn logit into label
    label = [key for key, value in indices.items() if value == np.argmax(prediction, axis=1)[0]]
    return label[0]
