import pytesseract
import cv2
import re



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def show_image_scaled(title, image, max_width=900):

   # Displays image resized to max width

    h, w = image.shape[:2]
   #Perserves aspect ratio
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(title, image)

#Run OCR on image
def extract_text_with_boxes(image, show_boxes=False, save_path=None, conf_threshold=40):
    #Filters with confidence score
    custom_cfg = "--oem 1 --psm 3 -c preserve_interword_spaces=1 --dpi 300"
    data = pytesseract.image_to_data(
        image, lang="ron+eng", config=custom_cfg,
        #gives text strings and confidence scores
        output_type=pytesseract.Output.DICT)
    #sets up the boxes
    boxed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    valid_lines = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        if conf <= 0:  # treat 0 / –1 as garbage
            continue
        if conf < conf_threshold:  # lower treshold
            continue

        if w < 10 or h < 10:#tiny boxes
            continue
        if not re.match(r'^[A-Za-z0-9ăâîșțĂÂÎȘȚ .:-]+$', text):#garbage chars
            continue
#if it passes, the word is added to this
        valid_lines.append(text)

        # bounding box
        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(boxed_image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if save_path:
        cv2.imwrite(save_path, boxed_image)

    if show_boxes:
        show_image_scaled('OCR Bounding Boxes', boxed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#returns all valid ocr word as a single string
    return "\n".join(valid_lines)
