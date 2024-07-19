# Nudity Detection

> "The only way to get rid of a temptation is to yield to it." - Oscar Wilde

## Introduction

This is a simple nudity detector that uses a custom-trained model to detect nudity in images. The model is trained on a dataset of 5,000 images, with 2,500 images of NSFW images and 2,500 images of non-NSFW images. The model is trained using a Convolutional Neural Network (CNN) and has an accuracy of 85%.

## Installation

To install, download the repository (or clone it) and run the following command:

```bash
pip install -r requirements.txt
```

## Usage

(See the [test.py](test.py) file for The complete explanation.)

### There are 3 ways to use the model:

- Predicting image, with no heatmap, and using the default model.
- Predicting image, with heatmap, and using the default model.
- Predicting image, with heatmap, and using a custom model.

### 1. Predicting image with no heatmap, and using the default model.

```python
from detect_nudity import NudityDetection

detector = NudityDetection()

image_path = "path/to/image.jpg"
result = detector.predict_image(image_path)

print(result)
```

### 2. Predicting image with heatmap, and using the default model.

```python
from detect_nudity import NudityDetection

detector = NudityDetection()

image_path = "path/to/image.jpg"
result = detector.predict_image(image_path, generate_heatmap=True)

print(result)
```

### 3. Predicting image with heatmap, and using a custom model.

```python
from detect_nudity import NudityDetection

detector = NudityDetection()

image_path = "path/to/image.jpg"
model_path = "path/to/model"

result = detector.predict_image(image_path, generate_heatmap=True, model_path=model_path)

print(result)
```

The `predict_image` method returns a tuple with the following elements:

- `is_nsfw`: A string indicating whether the image is NSFW or SFW.
- `percentage_nudity`: A float indicating the confidence score of the prediction.
- `hm_img`: A heatmap image showing the areas of the image that the model thinks are NSFW. (Only available if `generate_heatmap` is set to `True`.)

## Additional Information

- While running the model, the program checks if the model is present in the cache directory. If the model is not present, the program downloads the model from the Huggingface repo and saves it into the cache directory.
- Huggingface model: [nudity_detection](https://huggingface.co/esvinj312/nudity-detection)

## Fallacies

- The model is not perfect and may not be able to detect nudity in all images.
- The model may also generate false positives, i.e., it may detect nudity in images that do not contain nudity.
- The model currently detects nudity very slowly, taking around 1 to 3 seconds per image.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
