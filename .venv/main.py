import cv2
import pytesseract

from preprocessing import process_image
from ocr_engine import extract_text_with_boxes
from postprocessing import parse_fields
from roi_utils import extract_roi, ROI_MAP

#input image
pytesseract.pytesseract.tesseract_cmd = r'D:\New folder (3)\tesseract.exe';
#function defined to display the output image in a smaller format on the screen
def show_small(title, image, max_w=500):
    h, w = image.shape[:2]#take the image dimensions
    if w > max_w:
        scale = max_w / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))#rescale the window
    cv2.imshow(title, image)

if __name__ == '__main__':
    #we clean and prepare the image
    blurred, binary = process_image("now.jpg", show_steps=True)
  #  blurred, binary = process_image("romania_id_card.jpg", show_steps=True)

    #take the processed image and try to extract with ocr boxes from ocr_engine
    text = extract_text_with_boxes(binary, show_boxes=True, conf_threshold=70)
    fields = parse_fields(text)
  #print the extracted text
    print("OCR Text:")
    print(text)
    print("\nExtracted Fields:")
    print(fields)
#from ROI utils
    roi_results = {}
    #we iterate each field and its coordinates
    for field, (xr, yr, wr, hr) in ROI_MAP.items():
        roi = extract_roi(binary, xr, yr, wr, hr)
#psm 7 treats as single line
        config = "--psm 7 --oem 1"
#here we make sure we only extract digits for CNP
        if field == "cnp":
            config += " -c tessedit_char_whitelist=0123456789"

        result = pytesseract.image_to_string(roi, config=config, lang='ron').strip()
        if field == "cnp":
            result = ''.join(filter(str.isdigit, result))  # digits only

        roi_results[field] = result
        # visual debug, draw the ROI rectangle on a copy of the full image
        debug = binary.copy()  # or blurred.copy()
        #draw a box around the ROI zone
        H, W = debug.shape[:2]
        x1, y1 = int(xr * W), int(yr * H)
        x2, y2 = int((xr + wr) * W), int((yr + hr) * H)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        show_small("ROI-box " + field, debug, max_w=600)


    print("\nROI Extracted Fields:")
    for k, v in roi_results.items():
        print(f"{k:15}: {v}")


    cv2.waitKey(0)
    cv2.destroyAllWindows()