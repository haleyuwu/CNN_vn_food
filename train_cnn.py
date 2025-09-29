# train_cnn.py
import os, cv2, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ======== CONFIG ========
DATA_DIR    = "data/train"         # data/train/<class>/*.jpg
IMG_SIZE    = 64
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "cnn_vnfoods.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
TEST_SIZE   = 0.2
RANDOM_STATE= 42
EPOCHS      = 30
BATCH_SIZE  = 64
# ========================

os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset(data_dir, img_size):
    X, y = [], []
    for label in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, label)
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            img_path = os.path.join(p, f)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img.astype("float32")/255.0)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

def build_cnn(input_shape, num_classes):
    m = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def main():
    print(f"Loading dataset from: {DATA_DIR}")
    X, y = load_dataset(DATA_DIR, IMG_SIZE)
    print("Total images:", len(X))
    classes_sorted = sorted(list(set(y)))
    print("Classes:", classes_sorted)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)
    num_classes = y_cat.shape[1]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cat
    )

    model = build_cnn((IMG_SIZE, IMG_SIZE, 3), num_classes)
    model.summary()

    ckpt = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    es   = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)

    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[ckpt, es], verbose=1)

    model.save(MODEL_PATH)
    np.save(LABELS_PATH, le.classes_)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved labels to {LABELS_PATH}")

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
