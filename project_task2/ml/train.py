import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import pickle

TRAIN_DIR = "/mnt/c/Users/Asus/Desktop/datasets/fruit_classification/train/train"
TEST_DIR = "/mnt/c/Users/Asus/Desktop/datasets/fruit_classification/test/test"

IMAGE_SIZE = 96
NUM_CLASSES = 33
BATCH_SIZE = 128

IMAGE_GEN = ImageDataGenerator(rescale=1./255,
                              validation_split=0.2)

TRAIN_GEN = IMAGE_GEN.flow_from_directory(TRAIN_DIR, 
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                         batch_size=BATCH_SIZE,
                                          subset='training',
                                         class_mode='categorical')

TEST_GEN = IMAGE_GEN.flow_from_directory(TRAIN_DIR, 
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                         batch_size=BATCH_SIZE,
                                         subset='validation',
                                         class_mode='categorical')

INDICES = TRAIN_GEN.class_indices

class model:
    def __init__(self):
        self.model =None

    def create_model(self, image_size, num_classes):
        self.model= tf.keras.Sequential([
                Input(shape=(image_size, image_size, 3)),

                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),

                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),

                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),


                Flatten(),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
        print("Model created successfully")
        self.model.summary()
        

    def train_model(self, train_gen, test_gen, epochs=10):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_gen, validation_data=test_gen, epochs=epochs)
        self.model.save('/home/nasser/projects/project_task2/ml/model/fruit_classification.h5')
        print("Model trained successfully and saved")
    
