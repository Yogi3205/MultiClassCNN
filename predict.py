from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("multiclass_model.h5")

labels = ["cars", "human", "cats", "dogs"]

img = image.load_img("test_image.jpg", target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_index = np.argmax(prediction)

print("Prediction:", labels[class_index])