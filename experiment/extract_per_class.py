"""Extract per-class report for best PCA dim (60) using saved U-Net model."""
import numpy as np, cv2, os
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import warnings; warnings.filterwarnings('ignore')

IMG_SIZE = 256
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
CLASSES = ['Healthy', 'Common_Rust', 'Northern_Leaf_Blight', 'Gray_Leaf_Spot']

def load_dataset(data_dir):
    images, labels = [], []
    name_map = {'healthy': 0, 'Common_rust': 1, 'Northern_Leaf_Blight': 2, 'Cercospora_leaf_spot_Gray_leaf_spot': 3}
    for cls_name, cls_id in name_map.items():
        cls_path = os.path.join(data_dir, cls_name)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            for f in tqdm(files, desc=f'Load {cls_name}', leave=False):
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img); labels.append(cls_id)
    present_classes = [i for i in range(len(CLASSES)) if np.sum(np.array(labels)==i) > 0]
    present_names = [CLASSES[i] for i in present_classes]
    label_map = {old: new for new, old in enumerate(present_classes)}
    images_out, labels_out = [], []
    for img, lbl in zip(images, labels):
        if lbl in label_map: images_out.append(img); labels_out.append(label_map[lbl])
    return np.array(images_out), np.array(labels_out), present_names

def ds_combine(cls, d):
    counts = [np.sum(d == i) for i in range(len(np.unique(d)))]
    min_count = min(counts)
    keep = []
    for i in range(len(np.unique(d))):
        idx = np.where(d == i)[0]
        keep.extend(np.random.choice(idx, min_count, replace=False))
    return cls[keep], d[keep]

def extract_glcm_features(gray_img, distances, angles, props):
    features = []
    glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    for prop in props:
        if prop == 'ASM':
            energy = graycoprops(glcm, 'energy')
            features.extend(np.sqrt(energy).flatten())
        else:
            vals = graycoprops(glcm, prop)
            features.extend(vals.flatten())
    return np.array(features)

print("Loading saved U-Net...")
unet = tf.keras.models.load_model('experiment/models/unet_segmentation.keras')

print("Loading classification dataset...")
data_dir = os.path.expanduser('~/.keras/datasets/plantvillage_corn')
X, y, ACTIVE_CLASSES = load_dataset(data_dir)
X, y = ds_combine(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Segmenting...")
pred_masks = unet.predict(X_test/255.0, verbose=0)
pred_masks = (pred_masks > 0.5).astype(np.uint8).squeeze() * 255
X_test_seg = np.zeros_like(X_test)
for i in range(len(X_test)):
    mask_3ch = np.stack([pred_masks[i]/255.0]*3, axis=-1)
    X_test_seg[i] = (X_test[i] * mask_3ch).astype(np.uint8)

print("GLCM extraction...")
glcm_features = []
for img in tqdm(X_test_seg, desc='GLCM'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = extract_glcm_features(gray, DISTANCES, ANGLES, GLCM_PROPS)
    glcm_features.append(feats)
glcm_features = np.array(glcm_features)

print("PCA=60 classification...")
scaler = StandardScaler()
glcm_scaled = scaler.fit_transform(glcm_features)
pca = PCA(n_components=60)
X_feat = pca.fit_transform(glcm_scaled)
X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_test, test_size=0.2, random_state=42, stratify=y_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

cls_names = [c.replace('_', ' ') for c in ACTIVE_CLASSES]
print('\n=== Per-Class Report for PCA=60 ===')
print(classification_report(y_te, y_pred_rf, target_names=cls_names, digits=4))
acc = accuracy_score(y_te, y_pred_rf)
prec = precision_score(y_te, y_pred_rf, average='weighted')
rec = recall_score(y_te, y_pred_rf, average='weighted')
f1 = f1_score(y_te, y_pred_rf, average='weighted')
print(f'Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}')
