import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import h5py
import io

# Define the dice loss function
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator

# Load the trained model
def load_model_custom(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

model_path = r"C:\Users\User\Downloads\model2_save.h5"  # Use raw string for Windows path
model = load_model_custom(model_path)

# Function to preprocess images
def preprocess_image(image):
    img_array = np.zeros((1, 128, 128, 6))  # Prepare input array with 6 channels
    image = image.resize((128, 128))  # Resize image to 128x128
    img_array[0, :, :, :3] = np.array(image) / 255.0  # Normalize RGB channels
    img_array[0, :, :, 3:] = 0
    return img_array

def preprocess_h5_file(file_content):
    with h5py.File(io.BytesIO(file_content), 'r') as h5file:
        try:
            data = np.array(h5file['img'])  # Load the data from 'img'
        except KeyError as e:
            print(f"KeyError: {e}. Please check the dataset names in your .h5 file.")
            return None

    data[np.isnan(data)] = 0.000001
    img_array = np.zeros((1, 128, 128, 6))
    mid_rgb = data[:, :, 1:4].max() / 2.0
    mid_slope = data[:, :, 12].max() / 2.0
    mid_elevation = data[:, :, 13].max() / 2.0
    img_array[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb
    img_array[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb
    img_array[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb
    img_array[0, :, :, 3] = (data[:, :, 7] - data[:, :, 3]) / (data[:, :, 7] + data[:, :, 3] + 1e-6)
    img_array[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope
    img_array[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation
    return img_array

# Function to classify the image
def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    if model is None:
        messagebox.showerror("Error", "Model not loaded properly.")
        return
    
    img_array = None
    image = None  # Initialize image variable to avoid UnboundLocalError

    if file_path.endswith('.h5'):
        with open(file_path, 'rb') as f:
            file_content = f.read()
            img_array = preprocess_h5_file(file_content)
    else:
        try:
            image = Image.open(file_path).convert("RGB")  # Ensure the image is in RGB format
            img_array = preprocess_image(image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")
            return
    
    if img_array is None:
        messagebox.showerror("Error", "Error in preprocessing the image.")
        return
    
    prediction = model.predict(img_array)
    pred_mask = (prediction > 0.5).astype(np.uint8)
    landslide_pixels = np.sum(pred_mask)
    result = "Landslide detected!" if landslide_pixels > 0 else "No landslide detected."
    
    display_result(image, pred_mask, result)

# Function to display the results
def display_result(image, pred_mask, result):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    # Convert image to a NumPy array for display
    image_np = np.array(image)  # Convert the PIL image to a NumPy array

    # Original Image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Predicted Mask
    pred_mask_image = Image.fromarray((pred_mask[0, :, :, 0] * 255).astype(np.uint8))
    axes[1].imshow(pred_mask_image, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    # Overlay of Original Image and Mask
    axes[2].imshow(image_np)
    axes[2].imshow(pred_mask_image, alpha=0.5, cmap='jet')
    axes[2].set_title("Overlay of Mask on Image")
    axes[2].axis('off')
    
    plt.show()
    messagebox.showinfo("Result", result)

# Create the Tkinter GUI
root = tk.Tk()
root.title("Landslide Detection")

upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=20)

root.mainloop()
