import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Pre-Trained Model
print("Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights="imagenet")
print("Model loaded successfully!")

# Function to plot Bar Plot
def plot_bar(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=scores, y=labels, palette='viridis')
    plt.xlabel('Confidence (%)')
    plt.title('Bar Plot of Predictions')
    plt.tight_layout()
    plt.show()

# Function to plot Donut Chart
def plot_donut(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors, wedgeprops={'width':0.5})
    plt.title('Donut Chart of Predictions')
    plt.tight_layout()
    plt.show()

# Function to plot Scatter Plot
def plot_scatter(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    plt.scatter(labels, scores, color='red', s=100, edgecolors='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Confidence (%)')
    plt.title('Scatter Plot of Predictions')
    plt.tight_layout()
    plt.show()

# Step 2: Upload an Image or Capture Live Video
print("Choose an option:\n1. Upload a photo\n2. Use live video (press 'q' to quit)")
choice = input("Enter your choice (1/2): ")

if choice == "1":
    # Upload a photo
    print("Please upload a photo.")
    Tk().withdraw()  # Hide Tkinter root window
    image_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if not image_path:
        print("No image selected. Exiting...")
        exit()

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = preprocess_input(input_data)

    # Predict the species
    predictions = model.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")
    
    # Plot predictions separately
    plot_bar(decoded_predictions)
    plot_donut(decoded_predictions)
    plot_scatter(decoded_predictions)

elif choice == "2":
    # Use live video
    print("Starting live video. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_resized = cv2.resize(frame, (224, 224))
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = preprocess_input(input_data)

        predictions = model.predict(input_data)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        predicted_label, confidence = decoded_predictions[0][1], decoded_predictions[0][2] * 100

        cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Exiting...")
    exit()
