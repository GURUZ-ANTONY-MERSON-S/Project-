#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow matplotlib seaborn')


# 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('ls "/content/drive/MyDrive/kaggle/"')




# In[ ]:


import os, shutil, random

# Source folder with cloud categories
src_dir = "/content/drive/MyDrive/kaggle"
train_dir = "/content/drive/MyDrive/cloud_split/train"
test_dir = "/content/drive/MyDrive/cloud_split/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for category in os.listdir(src_dir):
    category_path = os.path.join(src_dir, category)
    if os.path.isdir(category_path):
        files = os.listdir(category_path)
        random.shuffle(files)
        split = int(0.8 * len(files))

        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        for i, file in enumerate(files):
            src = os.path.join(category_path, file)
            dst = os.path.join(train_dir if i < split else test_dir, category, file)
            shutil.copy(src, dst)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (128, 128)
batch_size = 32

train_path = "/content/drive/MyDrive/cloud_split/train"
test_path  = "/content/drive/MyDrive/cloud_split/test"

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)



# In[ ]:


import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[ ]:


# Function to check and clean directories
def clean_directory(directory):
    print(f"Checking and cleaning directory: {directory}")
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                filepath = os.path.join(subdir_path, filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Removing non-image file: {filepath}")
                    os.remove(filepath)
        else:
             # If it's a file directly in the main directory, remove it
            filepath = os.path.join(directory, subdir)
            if not subdir.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                 print(f"Removing non-image file: {filepath}")
                 os.remove(filepath)

# Clean the training and testing directories
clean_directory(train_path)
clean_directory(test_path)

# Recreate the image data generators after cleaning
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# In[ ]:


from PIL import Image
import os

def remove_corrupt_images(folder):
    total = 0
    removed = 0
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()  # verify image integrity
                total += 1
            except:
                os.remove(filepath)
                removed += 1
    print(f"Checked {total + removed} files. Removed {removed} corrupt images.")

# Clean both training and test folders
remove_corrupt_images("/content/drive/MyDrive/cloud_split/train")
remove_corrupt_images("/content/drive/MyDrive/cloud_split/test")


# In[ ]:


from PIL import Image
import os

def remove_corrupt_images(folder):
    total = 0
    removed = 0
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()  # verify image integrity
                total += 1
            except:
                os.remove(filepath)
                removed += 1
    print(f"Checked {total + removed} files. Removed {removed} corrupt images.")

# Clean both training and test folders
remove_corrupt_images("/content/drive/MyDrive/cloud_split/train")
remove_corrupt_images("/content/drive/MyDrive/cloud_split/test")

# Recreate the image data generators after cleaning
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# In[ ]:


import os

def remove_non_images(folder):
    allowed_exts = ['.jpg', '.jpeg', '.png']
    removed = 0
    for subdir, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in allowed_exts:
                filepath = os.path.join(subdir, file)
                os.remove(filepath)
                removed += 1
    print(f"Removed {removed} non-image files from: {folder}")

remove_non_images("/content/drive/MyDrive/cloud_split/train")
remove_non_images("/content/drive/MyDrive/cloud_split/test")


# In[ ]:


train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# In[ ]:


history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=5
)


# In[ ]:


import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Replace with your uploaded filename (check output of files.upload)
img_path = list(uploaded.keys())[0]

# Load image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_index = np.argmax(pred)
class_name = list(train_gen.class_indices.keys())[class_index]

# Show image and prediction
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {class_name}")
plt.show()

# Rain logic
rain_clouds = ['Cumulonimbus', 'Nimbostratus']
if class_name in rain_clouds:
    print("🌧️ It might rain.")
else:
    print("🌤️ It probably won't rain.")


# In[ ]:


get_ipython().system('pip install gradio')

import gradio as gr

def predict_cloud(img):
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_name = list(train_gen.class_indices.keys())[class_index]

    rain_clouds = ['Cumulonimbus', 'Nimbostratus']
    message = "🌧️ It might rain." if class_name in rain_clouds else "🌤️ It probably won't rain."
    return class_name, message

gr.Interface(fn=predict_cloud,
             inputs=gr.Image(type="pil"),
             outputs=["text", "text"]).launch()


# In[ ]:


model.save("cloud_classifier_model.keras")


# In[ ]:


from google.colab import files
files.download("cloud_classifier_model.keras")

