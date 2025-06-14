import cv2
import numpy as np
from math import atan2, degrees


# Card-edge detection + perspective warp
def find_card_quad(img):

   # Returns the 4 corner points of the largest rectangular contour
#Converts image to grayscale,blurs , detects edges using Canny.
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)        # relaxed Canny
    edges = cv2.dilate(edges, None, 1)      # close gaps
#Finds external contours, approx them to polygons
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
#calc area
    img_area = img.shape[0] * img.shape[1]
   #use top 10 largest contours
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        #calculates perimeter
        peri   = cv2.arcLength(c, True)
        #corner points that define the polygon
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #if not 4 point polygon, skip
        if len(approx) != 4:
            continue

        # aspect-ratio filter 1.6 for Romanian ID
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if not (1.55 <= ar <= 1.68):
            continue

        # area fraction filter 25 % of frame
        if cv2.contourArea(approx) < 0.25 * img_area:
            continue

        return approx.reshape(4, 2)
    return None

def _order_pts(pts):
    #tl, tr, br, bl in a order
    #matrix
    rect = np.zeros((4, 2), dtype="float32")
    #compute sum of x and y, min top left, max bot right
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    #same ideea but with diff for other diagonal
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def warp_card(img, quad, W=963, H=618):

    # Perspective warps the card quad to a flat canvas (W × H).
#takes the quad and orders them
    src = _order_pts(quad.astype("float32"))
    #tells each point where to go
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")

    #calculates the transf matrix M that maps he src to the rectangle dst.
    M   = cv2.getPerspectiveTransform(src, dst)
    #transformation M on the image img
    return cv2.warpPerspective(img, M, (W, H),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

def warp_full_frame(img, W=963, H=618):
   #Resize without found contour
    h, w = img.shape[:2]
   #gets the source image corners
    quad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    return warp_card(img, quad, W=W, H=H)


# NEW: simple white-balance

def white_balance_grayworld(img):

 #split in channgels
    b,g,r = cv2.split(img.astype(np.float32))
 #calc avr of each channel
    avg_b, avg_g, avg_r = [x.mean() for x in (b,g,r)]
    avg_gray = (avg_b + avg_g + avg_r) / 3
#scalin factors
    kb = avg_gray / avg_b
    kg = avg_gray / avg_g
    kr = avg_gray / avg_r
#scales each channel, equalize brightness
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
 #converts back to 8 bit int for display, processing.
    return cv2.merge([b,g,r]).astype(np.uint8)


def deskew_by_text_lines(img, max_angle=15, min_word_w=30):
  #convert to grayscale
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #then to binary
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#invert so text is white
  #connect chars that are close into blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    bw = cv2.dilate(bw, kernel, 1)
#fin word like shapes
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    angles = []
  #iterates blobs
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_word_w or h < 8:
            continue
            #filter small blobs
        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        #line trough blob and computes angle
        ang = degrees(np.arctan2(vy, vx))
        #angle should be in [-90 90] range
        if ang > 90:   ang -= 180
        if ang <= -90: ang += 180
        if abs(ang) <= max_angle:
            angles.append(ang)
#return original
    if not angles:
        return img, 0.0
#skip  if too small
    skew = np.median(angles)
    if abs(skew) < 0.1:
        return img, 0.0
#Creates a rotation matrix centered on the image

#Applies the rotation to correct skew
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), skew, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, skew

# deskewing using hough lines
def deskew_small(img, max_shear=10, hough_len_ratio=.5):

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#only lines longer than half the image width are considered
    min_len = int(gray.shape[1] * hough_len_ratio)
    #finds line segments in the edge image
    lines   = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                              minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return img, 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        #angle of each line with respect to horizontal
        ang = degrees(atan2(y2 - y1, x2 - x1))
        #ignore vertical or steep
        if -max_shear < ang < max_shear:
            angles.append(ang)

    if not angles:
        return img, 0.0

    skew = np.median(angles)
    if abs(skew) < 0.1:
        return img, 0.0

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, skew

#try hough first, then text
def deskew_hybrid(img):
    rot, ang = deskew_small(img, max_shear=10)
    if abs(ang) < 0.1:
        rot, ang = deskew_by_text_lines(rot)
    return rot, ang


def custom_grayscale(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray[i, j] = int(0.299*r + 0.587*g + 0.114*b)
    return gray

# gaussian blur
def apply_gaussian_blur(image, ksize=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)

# shows image resized
def show_image_scaled(title, image, max_width=900):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    cv2.imshow(title, image)

# otsu binarisation with safe tweak
def to_otsu_binary(gray, tweak=-13):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    base, _ = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #tresh_otsu calculates optimal treshold to separate text from background
    safe_thresh = int(max(base + tweak, 0.8 * base))
    _, binary = cv2.threshold(blur, safe_thresh, 255, cv2.THRESH_BINARY)
    return binary


def process_image(image_path, show_steps=False):
    raw = cv2.imread(image_path)

    # detect & warp card or whole frame
    quad = find_card_quad(raw)
    if quad is not None:
        raw = warp_card(raw, quad)       # perspective-correct
        if show_steps:
            show_image_scaled("Warped card (quad)", raw)
    else:
        raw = warp_full_frame(raw)       # uniform resize fallback
        if show_steps:
            show_image_scaled("Warped full frame", raw)

    #  white-balance
    raw = white_balance_grayworld(raw)
    if show_steps:
        show_image_scaled("White balanced", raw)

    # deskew
    deskewed, skew = deskew_hybrid(raw)
    if show_steps:
        show_image_scaled(f"Deskewed ({skew:+.2f}°)", deskewed)

    #  optional sharpening
    sharp = cv2.resize(deskewed, None,
                       fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # grayscale
    gray = custom_grayscale(sharp)
    if show_steps:
        show_image_scaled("Grayscale", gray)

    #  blur
    blurred = apply_gaussian_blur(gray, (3, 3))
    if show_steps:
        show_image_scaled("Blurred", blurred)

    #  Otsu binarisation
    binary = to_otsu_binary(blurred)           # tweak applied inside
    if show_steps:
        show_image_scaled("Binary (Otsu)", binary)

    # return both images for flexibility
    return blurred, binary
