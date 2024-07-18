# Nudity Detection

> "The only way to get rid of a temptation is to yield to it." - Oscar Wilde

## Introduction

This is a simple nudity detector that uses a pre-trained model to detect nudity in images. The model is trained on a dataset of 25,000 images, with 12,500 images of NSFW images and 12,500 images of non-NSFW images. The model is trained using a Convolutional Neural Network (CNN) and has an accuracy of 85%.

## Installation

To install the nudity detector, use the following command:

```bash
pip install nudity_detection
```

## Usage

```python
from nudity_detection import NudityDetection

detector = NudityDetection()

image_path = "path/to/image.jpg"
result = detector.predict_image(image_path)

print(result)
```

The `predict_image` method returns a tuple with the following elements:

- `is_nsfw`: A string indicating whether the image is NSFW or SFW.
- `percentage_nudity`: A float indicating the confidence score of the prediction.

An optional parameter `generate_heatmap` can be set to `True` to generate a heatmap of the image highlighting the areas that influenced the model the most in making the prediction.

```python
from nudity_detection import NudityDetection

detector = NudityDetection()

image_path = "path/to/image.jpg"
result = detector.predict_image(image_path, generate_heatmap=True)

print(result)
```

When `generate_heatmap` is `True`, the method returns a tuple with the following elements:

- `is_nsfw`: A string indicating whether the image is NSFW or SFW.
- `percentage_nudity`: A float indicating the confidence score of the prediction.
- `hm_img`: A string indicating the path to the generated heatmap image.

when `generate_heatmap` is `False`, the method returns a tuple with the following elements:

- `is_nsfw`: A string indicating whether the image is NSFW or SFW.
- `percentage_nudity`: A float indicating the confidence score of the prediction.
- `hm_img`: None

## Additional Information

- While running the model, the program checks if the model is present in the cache directory. If the model is not present, the program downloads the model from the Huggingface repo and saves it into the cache directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
