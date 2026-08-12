import cv2, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, Model, Input
import warnings; warnings.filterwarnings('ignore')

IMG_SIZE = 256
outdir = "images/seg_results"
os.makedirs(outdir, exist_ok=True)

sample_path = os.path.expanduser(
    "~/.keras/datasets/plantvillage_corn/Northern_Leaf_Blight/"
    "005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB 3853 copy.jpg")
img = cv2.imread(sample_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
cv2.imwrite(os.path.join(outdir, "a_original.jpg"), img)

# --- Otsu thresholding ---
def otsu_segment(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (25, 40, 40), (86, 255, 255))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(green_mask, otsu)
    kernel = np.ones((5,5), np.uint8)
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

# --- K-Means ---
def kmeans_segment(img_bgr, k=3):
    pixel_vals = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    green_ratio = np.array([c[1] / max(c[0]+c[2], 1) for c in centers])
    leaf_cluster = np.argmax(green_ratio)
    mask = (labels.reshape(img.shape[:2]) == leaf_cluster).astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
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

# --- U-Net ---
def build_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D(2)(c3)
    b = conv_block(p3, 256)
    u1 = layers.UpSampling2D(2)(b); u1 = layers.Concatenate()([u1, c3]); c4 = conv_block(u1, 128)
    u2 = layers.UpSampling2D(2)(c4); u2 = layers.Concatenate()([u2, c2]); c5 = conv_block(u2, 64)
    u3 = layers.UpSampling2D(2)(c5); u3 = layers.Concatenate()([u3, c1]); c6 = conv_block(u3, 32)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    return Model(inputs, outputs)

print("Training U-Net for segmentation demo...")
unet = build_unet()
unet.compile(optimizer='adam', loss='binary_crossentropy')
img_norm = img.astype(np.float32) / 255.0
mask_norm = (otsu_mask / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)
X_tr = np.stack([img_norm]*4, axis=0)
y_tr = np.stack([mask_norm[0]]*4, axis=0)
unet.fit(X_tr, y_tr, batch_size=2, epochs=20, verbose=0)
pred_mask = unet.predict(np.expand_dims(img_norm, 0), verbose=0)
pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze() * 255
unet_out = cv2.bitwise_and(img, img, mask=pred_mask)
cv2.imwrite(os.path.join(outdir, "d_unet.jpg"), unet_out)

print(f"Done: {os.listdir(outdir)}")
