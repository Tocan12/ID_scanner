import cv2

def extract_roi(image, x_ratio, y_ratio, w_ratio, h_ratio):
    #Crop region from image based on %
    h, w = image.shape[:2]
    # Convert to absolute pixel value
    x = int(x_ratio * w)
    y = int(y_ratio * h)
    w = int(w_ratio * w)
    h = int(h_ratio * h)
    # then slice the image
    return image[y:y + h, x:x + w]

# Interest zone definitions
ROI_MAP = {
    "cnp"         : (0.34, 0.22, 0.45, 0.06), #crop at 34% width, 22% height of image,box has 45% of width and 6% of size
    "nume"        : (0.30, 0.297, 0.29, 0.05),
    "prenume"     : (0.30, 0.357, 0.31, 0.05),
    "valabilitate": (0.69, 0.69, 0.30, 0.05),
    "mrz1"         : (0.05, 0.74, 0.94, 0.10),
    "mrz2": (0.05, 0.84, 0.94, 0.10),
    "loc nastere"  : (0.32, 0.507, 0.35, 0.05),
}
