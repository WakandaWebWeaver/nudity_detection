import os
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class NudityDetection:
    def __init__(self, model_repo='esvinj312/nudity-detection', model_filename='nude_detection_model.h5', target_size=(128, 128)):
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.target_size = target_size
        self.model_path = self.download_model()
        self.model = tf.keras.models.load_model(self.model_path)

    def download_model(self):
        if not os.path.exists(self.model_filename):
            print("Downloading model...")
            model_path = hf_hub_download(
                repo_id=self.model_repo, filename=self.model_filename)
            print(f"Model downloaded to {model_path}")
            return model_path
        else:
            print("Model already exists.")
            return self.model_filename

    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=self.target_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            return {'Error: ': 'Could not process image', 'File': 'main.py'}

    def predict_image(self, image_path, generate_heatmap=False):
        try:
            processed_image = self.preprocess_image(image_path)
            prediction = self.model.predict(processed_image)
            percentage_nudity = prediction[0][0] * 100
            is_nsfw = 'NSFW' if percentage_nudity > 50 else 'SFW'

            hm_img = None
            if generate_heatmap:
                hm_img = self.gen_heatmap(image_path)

            return is_nsfw, percentage_nudity, hm_img
        except Exception as e:
            return {'Error: ': 'Could not process image', 'File': 'main.py'}

    def gen_heatmap(self, image_path):
        try:
            img_array = self.get_img_array(image_path, size=self.target_size)
            heatmap = self.make_gradcam_heatmap(img_array)
            cam_path = self.save_and_display_gradcam(image_path, heatmap)
            return cam_path
        except Exception as e:
            return {'Error: ': 'Could not generate heatmap', 'File': 'heatmap.py'}

    def get_img_array(self, img_path, size):
        img = load_img(img_path, target_size=size)
        array = img_to_array(img) / 255.0
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_array):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(
                "conv2d_2").output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(
            pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def save_and_display_gradcam(self, img_path, heatmap, alpha=0.4):
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
        cam_path = f"{img_path.split('.')[0]}_heatmap.jpg"
        cv2.imwrite(cam_path, superimposed_img)
        return cam_path
