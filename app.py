from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter  # Import ImageFilter
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)

# Load trained model (Ensure the correct path)
model = tf.keras.models.load_model("trained_model.h5")

# Get class names from subdirectories in static/input_images/
CLASS_NAMES = sorted(os.listdir("static/input_images"))
print(CLASS_NAMES)  

# Image preprocessing function
def preprocessimage(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Ensure it matches model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array


# Create directories to store uploaded and processed images if they don't exist
INPUT_FOLDER = "static/input_images"
OUTPUT_FOLDER = "static/output_images"

UPLOAD_FOLDER = "static/input_images"
# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)
# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Initialize the model (this will be built dynamically)
def build_model(layers, num_classes):
    model = Sequential()

    # Start with a basic input layer (e.g., 64x64 images with 3 channels)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    
    # Add layers based on user selection
    for layer in layers:
        if layer['type'] == 'Conv2D':
            filters = int(layer['options']['filters'])
            activation = layer['options']['activation']
            model.add(Conv2D(filters, (3, 3), activation=activation))
        elif layer['type'] == 'MaxPooling2D':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif layer['type'] == 'Dense':
            units = int(layer['options']['units'])
            activation = layer['options']['activation']
            model.add(Dense(units, activation=activation))
        elif layer['type'] == 'Dropout':
            rate = float(layer['options']['rate'])
            model.add(Dropout(rate))

    # Flatten before passing to Dense layers
    model.add(Flatten())  # Flatten the feature maps to a 1D vector
    model.add(Dense(num_classes, activation='softmax'))  # Output layer based on number of classes

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model and return accuracy
def train_model(layers):
    # Preprocess images using ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Get the number of classes (subfolders in the directory)
    train_generator = datagen.flow_from_directory(
        OUTPUT_FOLDER, 
        target_size=(64, 64), 
        batch_size=32, 
        class_mode='categorical', 
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        OUTPUT_FOLDER, 
        target_size=(64, 64), 
        batch_size=32, 
        class_mode='categorical', 
        subset='validation'
    )

    # Get the number of classes dynamically
    num_classes = len(train_generator.class_indices)

    # Build and compile the model based on the number of classes
    model = build_model(layers, num_classes)

    # Train the model
    history = model.fit(train_generator, epochs=3, validation_data=val_generator)

    # Save the model
    model.save('trained_model.h5')

    # Get the final accuracy
    final_accuracy = history.history['accuracy'][-1]  # Last epoch training accuracy
    final_val_accuracy = history.history['val_accuracy'][-1]  # Last epoch validation accuracy

    return final_accuracy, final_val_accuracy


PREPROCESSING_METHODS = {
    "resize": "Resize",
    "grayscale": "Grayscale",
    "normalize": "Normalize",
    "histogram": "Histogram Equalization",
    "edge": "Edge Detection",
    "blur": "Gaussian Blur",
    "flip": "Flip Image",
    "rotate": "Rotate Image"
}

def preprocess_image(image, methods):
    if "grayscale" in methods:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "resize" in methods:
        image = cv2.resize(image, (128, 128))  # Resize to 128x128
    if "normalize" in methods:
        image = image / 255.0  # Normalize pixel values
    if "histogram" in methods and len(image.shape) == 2:  # Apply only if grayscale
        image = cv2.equalizeHist(image)
    if "edge" in methods:
        image = cv2.Canny(image, 100, 200)  # Apply edge detection
    if "blur" in methods:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    if "flip" in methods:
        image = cv2.flip(image, 1)  # Flip horizontally
    if "rotate" in methods:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
        image = cv2.warpAffine(image, M, (w, h))
    return image

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for preprocessing page
@app.route('/pre')
def pre():
     return render_template("preprocess.html", methods=PREPROCESSING_METHODS, processed_images={})

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == "POST":
        selected_methods = request.form.getlist("preprocessing")

        # Iterate through each labeled directory inside INPUT_FOLDER
        for label in os.listdir(INPUT_FOLDER):
            label_path = os.path.join(INPUT_FOLDER, label)
            
            if not os.path.isdir(label_path):  # Skip files, only process directories
                continue
            
            output_label_path = os.path.join(OUTPUT_FOLDER, label)  
            os.makedirs(output_label_path, exist_ok=True)  # Create output directory for label
            
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                image = cv2.imread(img_path)

                if image is None:
                    continue  # Skip invalid images

                processed_image = preprocess_image(image, selected_methods)
                
                # Save the processed image in the corresponding labeled directory
                output_path = os.path.join(output_label_path, filename)
                if len(processed_image.shape) == 2:  # Grayscale images
                    cv2.imwrite(output_path, processed_image)
                else:
                    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

        # Get list of processed images for display
        processed_images = {}
        for label in os.listdir(OUTPUT_FOLDER):
            label_path = os.path.join(OUTPUT_FOLDER, label)
            if os.path.isdir(label_path):
                processed_images[label] = os.listdir(label_path)  # Store filenames per label

        return render_template("preprocess.html", methods=PREPROCESSING_METHODS, processed_images=processed_images)

    return render_template("preprocess.html", methods=PREPROCESSING_METHODS, processed_images={})


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    label = request.form.get("label")  # Get label associated with the image

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the image in the appropriate label folder
    label_folder = os.path.join(UPLOAD_FOLDER, label)
    os.makedirs(label_folder, exist_ok=True)

    file_path = os.path.join(label_folder, file.filename)
    file.save(file_path)

    return jsonify({"message": f"Image '{file.filename}' uploaded successfully!"})


@app.route("/static/output_images/<folder>/<filename>")
def get_processed_image(folder, filename):
    folder_path = os.path.join(OUTPUT_FOLDER, folder)
    if os.path.exists(folder_path):
        return send_from_directory(folder_path, filename)
    else:
        return "Folder not found", 404
    
@app.route('/cnn')
def cnn():
    return render_template('cnn.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    layers = data['layers']  # Get the layers data from the request

    # Call train_model with the layers and get the accuracy
    train_acc, val_acc = train_model(layers)

    return jsonify({
        'message': 'Model trained and saved!',
        'train_accuracy': round(train_acc * 100, 2),  # Convert to percentage
        'validation_accuracy': round(val_acc * 100, 2)
    })

@app.route("/uploadimage", methods=["GET", "POST"])
def uploadimage():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("static/uploads", filename)
            file.save(filepath)  # Save the uploaded file

            # Preprocess and classify image
            img_array = preprocessimage(filepath)
            prediction = model.predict(img_array)
            print(prediction)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]

            return render_template("result.html", filename=filename, prediction=predicted_class)

    return render_template("upload.html")


if __name__ == '__main__':
    app.run(debug=True)
