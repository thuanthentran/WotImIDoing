import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# --- CẤU HÌNH ---
MODEL_PATH = 'best_slr_model.h5' 
SEQUENCE_LENGTH = 120            
NUM_FEATURES = 1755              
POSE_NOSE_INDEX = 0              
TOP_K = 4 # Số lượng dự đoán có xác suất cao nhất muốn hiển thị

# Tải ánh xạ nhãn (labels)
try:
    ACTIONS = sorted(os.listdir('Keypoints')) 
    if not ACTIONS:
        raise Exception("Thư mục Keypoints rỗng.")
except Exception as e:
    print(f"LỖI: Không thể tải danh sách nhãn. {e}")
    exit()

# --- KHỞI TẠO MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 1. TẢI VÀ CHUẨN BỊ MÔ HÌNH ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Đã tải mô hình thành công từ: {MODEL_PATH}")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình. Đảm bảo file {MODEL_PATH} tồn tại.")
    exit()

# --- 2. LOGIC TRÍCH XUẤT VÀ CHUẨN HÓA KEYPOINTS (Giữ nguyên) ---

def extract_and_normalize_keypoints(results):
    """
    Trích xuất và Chuẩn hóa Keypoints bằng Chuẩn hóa Kép.
    Phải KHỚP HOÀN TOÀN với logic huấn luyện.
    """
    pose_raw = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else None
    face_raw = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else None
    lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None
    rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else None

    if pose_raw is not None:
        nose_coords = pose_raw[POSE_NOSE_INDEX]
    else:
        nose_coords = np.zeros(3)

    def normalize_global(keypoint_array, size_if_none):
        if keypoint_array is not None:
            return (keypoint_array - nose_coords).flatten()
        return np.zeros(size_if_none * 3)
    
    face_global_norm = normalize_global(face_raw, 468)
    lh_global_norm = normalize_global(lh_raw, 21)
    rh_global_norm = normalize_global(rh_raw, 21)
    pose_global_norm = normalize_global(pose_raw, 33) 
    
    hand_shape_features = []
    
    if lh_raw is not None:
        lh_wrist = lh_raw[0] 
        lh_local_norm = (lh_raw - lh_wrist).flatten() 
        hand_shape_features.append(lh_local_norm)
    else:
        hand_shape_features.append(np.zeros(21 * 3))

    if rh_raw is not None:
        rh_wrist = rh_raw[0] 
        rh_local_norm = (rh_raw - rh_wrist).flatten()
        hand_shape_features.append(rh_local_norm)
    else:
        hand_shape_features.append(np.zeros(21 * 3))

    final_keypoints = np.concatenate([
        face_global_norm, lh_global_norm, rh_global_norm, pose_global_norm,
        hand_shape_features[0], hand_shape_features[1]  
    ])
    
    if final_keypoints.size != NUM_FEATURES:
         return np.zeros(NUM_FEATURES, dtype=np.float32) 
         
    return final_keypoints.astype(np.float32)

# ---------------------------------------------------------------------------------------
# HÀM CHÍNH: XỬ LÝ REAL-TIME
# ---------------------------------------------------------------------------------------

def recognize_sign_language():
    
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("LỖI: Không thể mở camera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        sequence = []           
        top_predictions = []    # Danh sách lưu trữ Top K dự đoán và xác suất
        
        print("\n--- Bắt đầu Nhận dạng Real-time (Hiển thị Top 4). Nhấn 'q' để thoát. ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) 

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- A. TRÍCH XUẤT KEYPOINTS VÀ TẠO CHUỖI ---
            keypoints = extract_and_normalize_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            # --- B. DỰ ĐOÁN KHI ĐỦ KHUNG HÌNH ---
            if len(sequence) == SEQUENCE_LENGTH:
                
                input_data = np.expand_dims(sequence, axis=0) 
                res = model.predict(input_data, verbose=0)[0]
                
                # SỬA ĐỔI: Lấy TOP K (4) kết quả
                sorted_indices = np.argsort(res)[-TOP_K:][::-1] # Sắp xếp giảm dần
                
                # Cập nhật danh sách Top K dự đoán
                top_predictions = []
                for i in sorted_indices:
                    top_predictions.append((ACTIONS[i], res[i]))
            
            # --- C. VẼ VÀ HIỂN THỊ TRỰC QUAN ---
            
            # Vẽ các điểm mốc
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS, 
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Hiển thị TOP K dự đoán
            if top_predictions:
                cv2.putText(image, "--- TOP 4 PREDICTIONS ---", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                for rank, (label, confidence) in enumerate(top_predictions):
                    # Màu xanh lá cho dự đoán cao nhất
                    color = (0, 255, 0) if rank == 0 else (0, 255, 255) 
                    text = f"{rank+1}. {label}: {confidence*100:.2f}%"
                    cv2.putText(image, text, (30, 80 + rank * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


            # Hiển thị độ dài chuỗi hiện tại
            cv2.putText(image, f"Frame Count: {len(sequence)}/{SEQUENCE_LENGTH}", (30, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Recognition Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    recognize_sign_language()