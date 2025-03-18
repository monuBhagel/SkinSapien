 
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
# dataset path defined

 
DATASET_DIR = "skin-disease-dataset"  # Fixed typo in dataset path
TRAIN_DIR = os.path.join(DATASET_DIR, "train_set")
TEST_DIR = os.path.join(DATASET_DIR, "test_set")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_data")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
 
# # Create directories for processed data

 
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_and_save_images(source_dir, output_dir):
    """Preprocess images by resizing and normalizing them."""
    os.makedirs(output_dir, exist_ok=True)
    
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        save_path = os.path.join(output_dir, category)
        os.makedirs(save_path, exist_ok=True)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Ensure only image files are processed
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip corrupted images
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # Normalize pixels
                cv2.imwrite(os.path.join(save_path, img_name), img * 255)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    print(f"Processed images saved in {output_dir}")
 
# # Preprocess Train and Test data

 
preprocess_and_save_images(TRAIN_DIR, os.path.join(PROCESSED_DIR, "train"))
preprocess_and_save_images(TEST_DIR, os.path.join(PROCESSED_DIR, "test"))
 
# # Image Data Generators

 
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator
 
# # Generate Data Loaders

 
train_gen, val_gen, test_gen = get_data_generators()
print("Data Preprocessing & Loading Complete!")


