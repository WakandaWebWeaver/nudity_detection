from detect_nudity import NudityDetection

detector = NudityDetection()

image_path = "image.jpeg"
result = detector.predict_image(image_path)

print(result)
