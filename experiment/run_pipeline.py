import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "experiment/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Healthy', 'Common_Rust', 'Northern_Leaf_Blight', 'Gray_Leaf_Spot']
IMG_SIZE = 256
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

def load_dataset(data_dir):
    """Load PlantVillage corn dataset from class directories"""
    images, labels = [], []
    name_map = {
        'healthy': 0,
        'Common_rust': 1,
        'Northern_Leaf_Blight': 2,
        'Cercospora_leaf_spot_Gray_leaf_spot': 3,
    }
    # First try class-directory structure (from fixed download)
    for cls_name, cls_id in name_map.items():
        cls_path = os.path.join(data_dir, cls_name)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f'  {CLASSES[cls_id]}: {len(files)} images')
            for f in tqdm(files, desc=f'Loading {cls_name}', leave=False):
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(cls_id)
    # Fallback: recursive search with filename matching
    if len(images) == 0:
        print('  No class directories found, trying recursive search...')
        name_map_fb = {
            'r.s_hl': 0, 'rs_glsp': 3, 'rs_nlb': 2,
            'Common_rust': 1, 'healthy': 0,
        }
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, f))
        print(f'  Found {len(all_files)} total images, classifying...')
        for fpath in tqdm(all_files, desc='Loading'):
            fname = os.path.basename(fpath).lower()
            cls = None
            for key, val in name_map_fb.items():
                if key.lower().replace('_', '') in fname.replace('_', ''):
                    cls = val
                    break
            if cls is None:
                continue
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(cls)
    
    for i, cname in enumerate(CLASSES):
        print(f'  {cname}: {np.sum(np.array(labels)==i)} images')

    # Filter out classes with 0 samples
    present_classes = [i for i in range(len(CLASSES)) if np.sum(np.array(labels)==i) > 0]
    present_names = [CLASSES[i] for i in present_classes]
    print(f'  Using classes: {present_names}')

    # Remap labels to be contiguous 0..n-1
    label_map = {old: new for new, old in enumerate(present_classes)}
    images_out, labels_out = [], []
    for img, lbl in zip(images, labels):
        if lbl in label_map:
            images_out.append(img)
            labels_out.append(label_map[lbl])
    return np.array(images_out), np.array(labels_out), present_names


def ds_combine(cls, d):
    """Downsample classes to balance dataset to minimum class count"""
    n_classes = len(np.unique(d))
    counts = [np.sum(d == i) for i in range(n_classes)]
    min_count = min(counts)
    print(f"  Balancing to {min_count} per class")
    keep = []
    for i in range(n_classes):
        idx = np.where(d == i)[0]
        keep.extend(np.random.choice(idx, min_count, replace=False))
    return cls[keep], d[keep]


def otsu_segment(img_bgr):
    """Segment leaf using Otsu thresholding (proxy for U-Net)"""
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


def build_unet():
    """Build lightweight U-Net for segmentation"""
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x
    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)
    # Bottleneck
    b = conv_block(p3, 256)
    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = conv_block(u1, 128)
    u2 = layers.UpSampling2D(2)(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 64)
    u3 = layers.UpSampling2D(2)(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = conv_block(u3, 32)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    return Model(inputs, outputs)


def extract_glcm_features(gray_img, distances, angles, props):
    """Extract GLCM features from grayscale image"""
    features = []
    glcm = graycomatrix(gray_img, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    for prop in props:
        if prop == 'ASM':
            energy = graycoprops(glcm, 'energy')
            features.extend(np.sqrt(energy).flatten())
        else:
            vals = graycoprops(glcm, prop)
            features.extend(vals.flatten())
    return np.array(features)


def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def main():
    print("="*60)
    print("Corn Disease Classification Pipeline")
    print("U-Net + GLCM + PCA + Random Forest")
    print("="*60)

    data_dir = os.path.expanduser("~/.keras/datasets/plantvillage_corn")
    os.makedirs(data_dir, exist_ok=True)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    X, y, ACTIVE_CLASSES = load_dataset(data_dir)
    X, y = ds_combine(X, y)
    print(f"  Total balanced samples: {len(X)} ({np.bincount(y).tolist()})")
    n_classes = len(ACTIVE_CLASSES)
    print(f"  Classes: {ACTIVE_CLASSES}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Segmentation
    print("\n[2/5] Segmentation (Otsu + U-Net)...")
    masks_train = np.zeros((len(X_train), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    masks_test = np.zeros((len(X_test), IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for i in tqdm(range(len(X_train)), desc="Seg train"):
        masks_train[i] = otsu_segment(X_train[i])
    for i in tqdm(range(len(X_test)), desc="Seg test"):
        masks_test[i] = otsu_segment(X_test[i])

    # Train lightweight U-Net
    unet = build_unet()
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m_train = masks_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    unet.fit(X_train / 255.0, m_train, batch_size=8, epochs=15, verbose=0,
             validation_split=0.2)

    # Apply U-Net segmentation
    pred_masks = unet.predict(X_test / 255.0, verbose=0)
    pred_masks = (pred_masks > 0.5).astype(np.uint8).squeeze() * 255

    # Segment images (apply mask)
    X_test_seg = np.zeros_like(X_test)
    for i in range(len(X_test)):
        mask_3ch = np.stack([pred_masks[i]/255.0]*3, axis=-1)
        X_test_seg[i] = (X_test[i] * mask_3ch).astype(np.uint8)

    # GLCM Feature Extraction
    print("\n[3/5] GLCM Feature Extraction...")
    glcm_features = []
    for img in tqdm(X_test_seg, desc="GLCM"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = extract_glcm_features(gray, DISTANCES, ANGLES, GLCM_PROPS)
        glcm_features.append(feats)
    glcm_features = np.array(glcm_features)
    print(f"  GLCM feature shape: {glcm_features.shape}")

    # PCA Feature Selection
    print("\n[4/5] PCA Feature Selection...")
    scaler = StandardScaler()
    glcm_scaled = scaler.fit_transform(glcm_features)

    pca_dims = [10, 30, 60, 90, 120]
    rf_results = {}
    svm_results = {}

    for dim in pca_dims:
        if dim < glcm_features.shape[1]:
            pca = PCA(n_components=min(dim, glcm_features.shape[0]))
            X_feat = pca.fit_transform(glcm_scaled)
        else:
            X_feat = glcm_scaled.copy()

        X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_test, test_size=0.2,
                                                    random_state=42, stratify=y_test)

        # Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_te)
        t_rf = time.time() - t0

        rf_results[dim] = {
            'accuracy': accuracy_score(y_te, y_pred_rf),
            'precision': precision_score(y_te, y_pred_rf, average='weighted'),
            'recall': recall_score(y_te, y_pred_rf, average='weighted'),
            'f1': f1_score(y_te, y_pred_rf, average='weighted'),
            'time': t_rf,
            'confusion': confusion_matrix(y_te, y_pred_rf),
            'pred': y_pred_rf,
            'true': y_te,
        }

        # SVM Baseline
        t0 = time.time()
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_tr, y_tr)
        y_pred_svm = svm.predict(X_te)
        t_svm = time.time() - t0

        svm_results[dim] = {
            'accuracy': accuracy_score(y_te, y_pred_svm),
            'precision': precision_score(y_te, y_pred_svm, average='weighted'),
            'recall': recall_score(y_te, y_pred_svm, average='weighted'),
            'f1': f1_score(y_te, y_pred_svm, average='weighted'),
            'time': t_svm,
            'confusion': confusion_matrix(y_te, y_pred_svm),
            'pred': y_pred_svm,
            'true': y_te,
        }

        print(f"  PCA={dim:3d}: RF acc={rf_results[dim]['accuracy']:.4f} "
              f"({t_rf:.2f}s) | SVM acc={svm_results[dim]['accuracy']:.4f} "
              f"({t_svm:.2f}s)")

    # CNN Baseline (on raw images)
    print("\n  CNN Baseline (ConvNet on raw images)...")
    t0 = time.time()
    cnn = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    cnn.fit(X_train / 255.0, y_train, batch_size=16, epochs=25, verbose=0,
            validation_split=0.2)
    y_pred_cnn = np.argmax(cnn.predict(X_test / 255.0, verbose=0), axis=1)
    t_cnn = time.time() - t0
    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    cnn_cm = confusion_matrix(y_test, y_pred_cnn)
    print(f"  CNN: accuracy={cnn_acc:.4f} ({t_cnn:.1f}s total)")

    # Generate Results
    print("\n[5/5] Generating Results...")

    # Table 1: RF per PCA dim
    print("\n  --- RF Results per PCA dimension ---")
    print(f"  {'Dim':>5}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Time(s)':>7}")
    print("  " + "-"*58)
    for dim in pca_dims:
        r = rf_results[dim]
        print(f"  {dim:>5}  {r['accuracy']:>8.4f}  {r['precision']:>9.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['time']:>7.2f}")

    # Table 2: Method Comparison
    print("\n  --- Method Comparison (best PCA dim) ---")
    r = rf_results[120]
    s = svm_results[120]
    print(f"  {'Method':<20} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Time(s)':>7}")
    print("  " + "-"*65)
    print(f"  {'Random Forest (GLCM)':<20} {r['accuracy']:>8.4f} {r['precision']:>9.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['time']:>7.2f}")
    print(f"  {'SVM (GLCM)':<20} {s['accuracy']:>8.4f} {s['precision']:>9.4f} "
          f"{s['recall']:>7.4f} {s['f1']:>7.4f} {s['time']:>7.2f}")
    print(f"  {'CNN (Raw Images)':<20} {cnn_acc:>8.4f} {'N/A':>9} {'N/A':>7} {'N/A':>7} {t_cnn:>7.1f}")

    # Confusion Matrices
    cls_names = [c.replace('_', ' ') for c in ACTIVE_CLASSES]
    plot_confusion_matrix(r['confusion'], cls_names,
                          'Random Forest (120 GLCM features)', 'cm_rf.png')
    plot_confusion_matrix(s['confusion'], cls_names,
                          'SVM (120 GLCM features)', 'cm_svm.png')
    plot_confusion_matrix(cnn_cm, cls_names,
                          'CNN (Raw Images)', 'cm_cnn.png')

    # Per-class report for best RF
    print("\n  --- Per-Class Report (RF, 120 features) ---")
    print(classification_report(r['true'], r['pred'], target_names=cls_names))

    # Time computation plot
    times = [rf_results[d]['time'] for d in pca_dims]
    plt.figure(figsize=(8, 5))
    plt.bar([str(d) for d in pca_dims], times, color='steelblue')
    plt.xlabel('Number of PCA dimensions of features')
    plt.ylabel('Time computation (s)')
    plt.title('Average computation time for each number of PCA dimensions')
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f'{v:.1f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_computation.png'), dpi=150)
    plt.close()

    # Accuracy per PCA dim plot
    accs = [rf_results[d]['accuracy'] for d in pca_dims]
    plt.figure(figsize=(8, 5))
    plt.plot(pca_dims, accs, 'o-', color='darkgreen', linewidth=2, markersize=8)
    plt.xlabel('Number of PCA dimensions')
    plt.ylabel('Accuracy')
    plt.title('RF Accuracy vs PCA dimensions')
    plt.ylim(0.7, 1.0)
    for i, v in enumerate(accs):
        plt.text(pca_dims[i], v + 0.005, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_pca.png'), dpi=150)
    plt.close()

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
