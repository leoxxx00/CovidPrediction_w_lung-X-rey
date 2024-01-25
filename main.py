import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageTk
import numpy as np

# Load your trained model
MODEL_PATH = 'Detection_Covid_19.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary
print('Model loaded.')

class CovidPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Covid-19 Predictor")

        # Create labels
        self.label = tk.Label(root, text="Upload an image:")
        self.label.pack()

        # Create buttons
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_image)
        self.predict_button.pack()

        # Create result labels
        self.result_label1 = tk.Label(root, text="")
        self.result_label1.pack()

        self.result_label2 = tk.Label(root, text="")
        self.result_label2.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])

        self.image_path = file_path
        self.display_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo

    def predict_image(self):
        if hasattr(self, 'image_path'):
            prediction = model_predict(self.image_path, model)
            result = 'Positive For Covid-19' if prediction == 0 else 'Negative for Covid-19'
            self.result_label1.config(text=f'Result: {result}')
        else:
            self.result_label1.config(text="Please upload an image first.")

def model_predict(img_path, model):
    xtest_image = Image.open(img_path).convert('RGB')
    xtest_image = xtest_image.resize((224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis=0)
    preds = model.predict(xtest_image)
    return preds[0][0]

if __name__ == '__main__':
    root = tk.Tk()
    app = CovidPredictorGUI(root)
    root.mainloop()
