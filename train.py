import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# --- CẤU HÌNH DỰ ÁN ---
KEYPOINTS_ROOT = 'Keypoints'       # Thư mục chứa các file .npy đã trích xuất
MAX_FRAMES = 120                   # Độ dài chuỗi (Sequence Length)
NUM_FEATURES = 1755                # Số lượng đặc trưng (feature count) đã tối ưu
RANDOM_STATE = 42                  # Để đảm bảo khả năng tái tạo kết quả

# --------------------------------------------------------------------
# GIAI ĐOẠN 2: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# --------------------------------------------------------------------

def load_and_preprocess_data():
    """Tải các file .npy, Pad chuỗi, và chia tập Train/Test/Validation."""
    print("\n--- Bắt đầu GIAI ĐOẠN 2: Tải và Tiền xử lý Dữ liệu ---")
    
    # 1. Lấy danh sách nhãn và tạo ánh xạ
    actions = sorted(os.listdir(KEYPOINTS_ROOT))
    if not actions:
        raise FileNotFoundError("Không tìm thấy thư mục keypoints. Hãy chạy extract_data.py trước.")
        
    label_map = {label: num for num, label in enumerate(actions)}
    NUM_CLASSES = len(actions)
    print(f"Tìm thấy {NUM_CLASSES} nhãn (lớp).")

    sequences, labels = [], []
    
    # 2. Tải dữ liệu từ file .npy
    for action in actions:
        action_path = os.path.join(KEYPOINTS_ROOT, action, '*.npy')
        for npy_file in glob.glob(action_path):
            try:
                # Tải dữ liệu đã được lưu dưới dạng float32
                sequence = np.load(npy_file) 
                sequences.append(sequence)
                labels.append(label_map[action])
            except Exception as e:
                print(f"Lỗi khi tải {npy_file}: {e}")

    # 3. Padding chuỗi: Đồng bộ về độ dài MAX_FRAMES
    print(f"Tổng cộng {len(sequences)} mẫu được tải. Bắt đầu Padding...")
    X = pad_sequences(sequences, maxlen=MAX_FRAMES, padding='post', dtype='float32')

    # 4. One-Hot Encoding cho nhãn
    y = to_categorical(labels).astype(int)

    # 5. Chia tập huấn luyện (Train), kiểm tra (Validation), và đánh giá cuối cùng (Test)
    # Tách 90% cho Train/Val, 10% cho Test cuối cùng (đánh giá khách quan)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y
    )
    # Tách tiếp 80% Train và 20% Validation từ X_train_val (tương đương 72% Train, 18% Val trên tổng)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.20/0.90), random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"Shape Dữ liệu (X): {X.shape}")
    print(f"Phân chia: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES

# --------------------------------------------------------------------
# GIAI ĐOẠN 3: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH LSTM
# --------------------------------------------------------------------

def build_slr_model(num_classes):
    """Thiết kế kiến trúc mô hình LSTM cho nhận dạng ngôn ngữ ký hiệu."""

    model = Sequential()
    
    # 1. Lớp Masking: Rất quan trọng để bỏ qua các giá trị 0 (padding)
    model.add(Masking(mask_value=0., input_shape=(MAX_FRAMES, NUM_FEATURES)))

    # 2. Lớp LSTM đầu tiên: Học mẫu chuyển động
    model.add(LSTM(128, return_sequences=True, activation='tanh', 
                   kernel_initializer='he_normal', # Khởi tạo trọng số tốt hơn
                   recurrent_initializer='orthogonal'))
    model.add(Dropout(0.3)) # Tăng dropout để chống overfitting

    # 3. Lớp LSTM thứ hai: Học các đặc trưng phức tạp hơn
    model.add(LSTM(256, return_sequences=False, activation='tanh')) 
    model.add(Dropout(0.3))

    # 4. Lớp Dense: Phân loại
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # 5. Lớp Output: Phân loại đa lớp
    model.add(Dense(num_classes, activation='softmax'))

    # Biên dịch mô hình với tối ưu hóa Adam
    model.compile(
        optimizer=Adam(learning_rate=0.00001, clipnorm=1.0), 
        loss='categorical_crossentropy', 
        metrics=['categorical_accuracy']
    )
    
    return model

def train_slr_model(X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES):
    """Thực hiện huấn luyện mô hình với Callbacks."""
    
    model = build_slr_model(NUM_CLASSES)
    model.summary()

    # Thiết lập Callbacks:
    
    # 1. Early Stopping: Dừng sớm nếu độ chính xác trên Validation không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy', 
        patience=15, # Chờ 15 epoch trước khi dừng
        verbose=1, 
        mode='max',
        restore_best_weights=True # Tải lại trọng số tốt nhất
    ) 

    # 2. Model Checkpoint: Lưu mô hình tốt nhất
    checkpoint = ModelCheckpoint(
        'best_slr_model.h5', 
        monitor='val_categorical_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )

    # Huấn luyện
    print("\n--- Bắt đầu Huấn luyện Mô hình ---")
    history = model.fit(
        X_train, y_train, 
        epochs=100, # Đặt số epoch cao, Early Stopping sẽ quản lý việc dừng
        batch_size=32, 
        validation_data=(X_val, y_val), 
        callbacks=[early_stopping, checkpoint]
    )
    
    # Đánh giá cuối cùng trên tập Test (chưa bao giờ được mô hình nhìn thấy)
    print("\n--- Đánh giá cuối cùng trên tập Test ---")
    # Tải lại mô hình tốt nhất từ Checkpoint để đánh giá
    best_model = tf.keras.models.load_model('best_slr_model.h5')
    loss, acc = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"\nĐộ chính xác Cuối cùng trên tập Test: {acc*100:.2f}%")

# --- CHẠY CHƯƠNG TRÌNH CHÍNH ---
if __name__ == '__main__':
    # Đảm bảo sử dụng GPU nếu có
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Chỉ sử dụng VRAM khi cần thiết (giúp tránh lỗi Out of Memory)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Đang sử dụng GPU cho quá trình huấn luyện.")
        except RuntimeError as e:
            print(e)

    X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES = load_and_preprocess_data()
    
    train_slr_model(X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES)