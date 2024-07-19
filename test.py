from detect_nudity import NudityDetection

# Step 1: Create an instance of the NudityDetection class
detector = NudityDetection()

# Step 2: Define the path to the image you want to predict
image_path = "image.jpeg"

# Assign the result of the prediction to a variable, we are not generating a heatmap
# For heatmap, set generate_heatmap=True, to use a custom model, pass the model path as an argument

result = detector.predict_image(image_path)

# Example of using a custom model path
# result = detector.predict_image(image_path, model_path="path/to/custom/model.h5")

# Step 3: Print the result
print(result)


# Alternate methods to use the class

# # Method 1: Not initializing the class
# result = NudityDetection().predict_image(image_path)
# print(result)

# # Method 2: Using the class as a context manager
# with NudityDetection() as detector:
#     result = detector.predict_image(image_path)
#     print(result)

# # Method 3: Using the class as a context manager with a custom model path
# model_path = "path/to/custom/model.h5"

# with NudityDetection(model_filename=model_path) as detector:
#     result = detector.predict_image(image_path)
#     print(result)

# # Method 4: Using the class as a context manager with a custom model path and target size
# model_path = "path/to/custom/model.h5"

# with NudityDetection(model_filename=model_path, target_size=(256, 256)) as detector:
#     result = detector.predict_image(image_path)
#     print(result)
