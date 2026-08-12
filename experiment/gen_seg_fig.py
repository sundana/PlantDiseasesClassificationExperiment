"""Generate segmentation comparison figure using the properly trained U-Net model."""
import cv2, numpy as np, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import warnings; warnings.filterwarnings('ignore')

IMG_SIZE = 256
outdir = "images/seg_results"
os.makedirs(outdir, exist_ok=True)

# Use the same sample leaf as before (Northern Leaf Blight)
sample_path = os.path.expanduser(
    "~/.keras/datasets/plantvillage_corn/Northern_Leaf_Blight/"
    "005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB 3853 copy.jpg")
img = cv2.imread(sample_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
h, w = img.shape[:2]
cv2.imwrite(os.path.join(outdir, "a_original.jpg"), img)

# --- K-Means segmentation ---
def kmeans_segment(img_bgr, k=3):
    pixel_vals = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    green_ratio = np.array([c[1] / max(c[0] + c[2], 1) for c in centers])
    leaf_cluster = np.argmax(green_ratio)
    mask = (labels.reshape(img.shape[:2]) == leaf_cluster).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)

kmeans_out = kmeans_segment(img)
cv2.imwrite(os.path.join(outdir, "c_kmeans.jpg"), kmeans_out)

# --- Otsu thresholding ---
def otsu_segment(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (25, 40, 40), (86, 255, 255))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(green_mask, otsu)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

otsu_mask = otsu_segment(img)
otsu_out = cv2.bitwise_and(img, img, mask=otsu_mask)
cv2.imwrite(os.path.join(outdir, "b_otsu.jpg"), otsu_out)

# --- U-Net (properly trained on COCO annotations) ---
print("Loading trained U-Net model...")
unet = tf.keras.models.load_model("experiment/models/unet_segmentation.keras")

img_norm = img.astype(np.float32) / 255.0
pred_mask = unet.predict(np.expand_dims(img_norm, 0), verbose=0)
pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze() * 255
unet_out = cv2.bitwise_and(img, img, mask=pred_mask)
cv2.imwrite(os.path.join(outdir, "d_unet.jpg"), unet_out)

print(f"Done: {os.listdir(outdir)}")
print("U-Net segmentation mask mean:", pred_mask.mean())
