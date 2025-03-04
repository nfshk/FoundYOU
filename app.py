from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import os
from datetime import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import base64


app = Flask(__name__)
captured_images_folder = 'static/captured_images'

if not os.path.exists(captured_images_folder):
    os.makedirs(captured_images_folder)

dataset = pd.read_csv("static/dataset_foundyou.csv")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([char * 2 for char in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def recommend_foundation(hex_color, top_n=3):
    # Convert hex color to RGB
    rgb_color = np.array(hex_to_rgb(hex_color)).reshape(1, -1)

    # Use KMeans to cluster the dataset into color groups
    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
    dataset['cluster'] = kmeans.fit_predict(dataset[['red', 'green', 'blue']])

    # Find the cluster of the input color
    input_cluster = kmeans.predict(rgb_color)[0]

    # Filter dataset to include only products in the same cluster
    cluster_dataset = dataset[dataset['cluster'] == input_cluster]

    # Calculate Euclidean distance for each row in the filtered dataset
    cluster_dataset['distance'] = cluster_dataset.apply(lambda row: np.linalg.norm(np.array(row[['red', 'green', 'blue']]) - np.array(rgb_color)), axis=1)

    # Sort the filtered dataset based on distance
    sorted_cluster_dataset = cluster_dataset.sort_values(by='distance')

    # Select the top N recommendations
    recommendations = sorted_cluster_dataset.head(top_n)

    return recommendations[['brand', 'product', 'description', 'url', 'hex', 'current_source']]


dataset[['red', 'green', 'blue']] = pd.DataFrame([hex_to_rgb(hex_color) for hex_color in dataset['hex']])

def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = img[y:y+h, x:x+w]
        return face_roi
    else:
        return None
    
def detect_skin_color(face_image):
    # Ambil nilai piksel BGR pada gambar wajah
    pixels = face_image.reshape((-1, 3))

    # Inisialisasi model GMM dengan 3 komponen (sesuaikan jika diperlukan)
    gmm = GaussianMixture(n_components=3, random_state=42)

    # Fitting model pada data piksel warna
    gmm.fit(pixels)

    # Ambil rata-rata warna dari pusat kluster dengan bobot tertinggi
    skin_color = gmm.means_[np.argmax(gmm.weights_)]

    # Ambil nilai maksimum dari r, g, b untuk mempertahankan kecerahan warna
    skin_color_max = np.max(skin_color)

    # Normalisasi nilai warna kulit agar kecerahan tetap terjaga
    normalized_skin_color = (skin_color / skin_color_max) * 255

    # Representasi warna kulit dalam format HEX
    skin_color_hex = rgb_to_hex(normalized_skin_color)

    return skin_color_hex

def rgb_to_hex(rgb):
    # Pastikan nilai RGB berada dalam rentang 0-255
    rgb = np.clip(rgb, 0, 255)

    # Format nilai RGB sebagai string HEX
    return "#{:02X}{:02X}{:02X}".format(int(rgb[2]), int(rgb[1]), int(rgb[0]))

def hsv_to_hex(h, s, v):
    rgb_color = np.array([h, s, v], dtype=np.uint8).reshape(1, 1, 3)
    hex_color = cv2.cvtColor(rgb_color, cv2.COLOR_HSV2RGB)[0, 0]
    hex_string = "#{:02x}{:02x}{:02x}".format(hex_color[0], hex_color[1], hex_color[2])
    return hex_string

@app.route('/')
def index():
    return render_template('main.html')


@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Check if 'image_data' is in the request form
        if 'image_data' not in request.form:
            return jsonify({'status': 'failed', 'message': 'No image data'}), 400

        # Get the base64-encoded image data from the form
        image_data = request.form['image_data']

        # Decode base64 and save the image
        decoded_image_data = base64.b64decode(image_data)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'image_{timestamp}.jpg'
        file_path = os.path.join(captured_images_folder, filename)

        with open(file_path, 'wb') as f:
            f.write(decoded_image_data)

        face_image = detect_face(file_path)
        if face_image is not None:
            skin_color_hex = detect_skin_color(face_image)
            return jsonify({'status': 'success', 'image_path': file_path, 'skin_color': skin_color_hex})
        else:
            return jsonify({'status': 'failed', 'message': 'No face detected'}), 400

    except Exception as e:
        app.logger.error(f"Error in /capture route: {str(e)}")
        return jsonify({'status': 'failed', 'message': 'Internal Server Error'}), 500

@app.route('/display_captured')
def display_captured():
    images = [f'{captured_images_folder}/{img}' for img in os.listdir(captured_images_folder)]
    return render_template('display_captured.html', images=images)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'failed', 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'failed', 'message': 'No selected file'})

    if file:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'image_{timestamp}.jpg'
        file_path = os.path.join(captured_images_folder, filename)
        file.save(file_path)

        face_image = detect_face(file_path)
        if face_image is not None:
            skin_color_hex = detect_skin_color(face_image)
            return jsonify({'status': 'success', 'image_path': file_path, 'skin_color': skin_color_hex})
        else:
            return jsonify({'status': 'failed', 'message': 'No face detected'})
    else:
        return jsonify({'status': 'failed', 'message': 'Error uploading file'})

@app.route('/recommendation')
def get_recommendation():
    try:
        hex_color = request.args.get('hex_color')
    
        # Call the recommend_foundation function with the provided hex_color
        foundation_recommendation = recommend_foundation(hex_color)
    
        # Convert DataFrame to dictionary
        recommendation_dict = foundation_recommendation.to_dict(orient='records')[0]
    
        return jsonify({
            'status': 'success',
            'foundation_recommendation': recommendation_dict
        })
    except Exception as e:
        # Log the exception
        app.logger.error(f"Error in /recommendation route: {str(e)}")
        return jsonify({'status': 'failed', 'message': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
