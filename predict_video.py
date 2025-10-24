import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'best_slr_model.h5'       # File mô hình đã huấn luyện
INPUT_VIDEO_PATH = 'testvideo.mp4'    # <<< THAY THẾ bằng đường dẫn video của bạn
OUTPUT_VIDEO_PATH = 'output_predict.mp4' # Tên file video đầu ra
SEQUENCE_LENGTH = 120                  # Độ dài chuỗi (MAX_FRAMES)
NUM_FEATURES = 1755                    # Số lượng đặc trưng
POSE_NOSE_INDEX = 0                    
TOP_K = 4                              # Số lượng dự đoán có xác suất cao nhất

# Tải ánh xạ nhãn (labels)
try:
    ACTIONS = sorted(os.listdir('Keypoints')) 
    if not ACTIONS:
        raise Exception("Thư mục Keypoints rỗng.")
except Exception as e:
    print(f"LỖI: Không thể tải danh sách nhãn. {e}")
    exit()

# --- KHỞI TẠO VÀ TẢI MÔ HÌNH ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"LỖI: Không thể tải mô hình. {e}")
    exit()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- LOGIC TRÍCH XUẤT VÀ CHUẨN HÓA KEYPOINTS (Giữ nguyên) ---
# Hàm này phải khớp hoàn toàn với logic huấn luyện
def extract_and_normalize_keypoints(results):
    # ... (Giữ nguyên logic extract_and_normalize_keypoints từ realtime_recognition.py) ...
    # Để tránh lặp lại code, hãy đảm bảo bạn SAO CHÉP hàm này từ realtime_recognition.py vào đây.
    
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
# HÀM CHÍNH: XỬ LÝ VIDEO FILE
# ---------------------------------------------------------------------------------------

def predict_on_video():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video file tại {INPUT_VIDEO_PATH}")
        return

    # Lấy thông số video đầu vào
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Khởi tạo VideoWriter để lưu video đầu ra
    # Cần đảm bảo codec 'mp4v' hoạt động trên hệ thống của bạn
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        sequence = []           
        frame_count = 0
        
        print(f"--- Bắt đầu xử lý video: {INPUT_VIDEO_PATH} ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            # Xử lý hình ảnh (không cần lật frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- A. TRÍCH XUẤT KEYPOINTS VÀ CẬP NHẬT CHUỖI ---
            keypoints = extract_and_normalize_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            top_predictions = []
            
            # --- B. DỰ ĐOÁN KHI ĐỦ KHUNG HÌNH (Sử dụng cửa sổ trượt) ---
            if len(sequence) >= 5: # Thử dự đoán ngay khi có ít nhất 5 khung hình
                
                # CHỈNH SỬA QUAN TRỌNG: Áp dụng Padding ngay trước dự đoán
                # 1. Chuyển list sequence thành NumPy array
                current_sequence = np.array(sequence) 
                
                # 2. Tạo một mảng padding (ví dụ: mảng zero)
                # Tính số lượng zero cần thêm
                pad_length = SEQUENCE_LENGTH - len(current_sequence)
                
                # 3. Padding: Thêm zero vào CUỐI chuỗi để đạt đủ 120
                padded_sequence = np.pad(
                    current_sequence, 
                    ((0, pad_length), (0, 0)), 
                    mode='constant', 
                    constant_values=0
                )
                
                # 4. Biến đổi chuỗi thành tensor đầu vào (1, SEQUENCE_LENGTH, NUM_FEATURES)
                input_data = np.expand_dims(padded_sequence, axis=0) 
                
                # Dự đoán
                res = model.predict(input_data, verbose=0)[0]
                
                # Lấy TOP K (4) kết quả
                sorted_indices = np.argsort(res)[-TOP_K:][::-1] 
                
                # Lưu Top K dự đoán
                for i in sorted_indices:
                    top_predictions.append((ACTIONS[i], res[i]))
            
            # --- C. VẼ VÀ GHI KẾT QUẢ VÀO VIDEO ---
            
            # Vẽ các điểm mốc
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Hiển thị TOP K dự đoán
            if top_predictions:
                cv2.putText(image, "--- TOP 4 PREDICTIONS ---", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                for rank, (label, confidence) in enumerate(top_predictions):
                    color = (0, 255, 0) if rank == 0 else (0, 255, 255) 
                    text = f"{rank+1}. {label}: {confidence*100:.2f}%"
                    cv2.putText(image, text, (30, 80 + rank * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(image, "Collecting frames...", (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


            # Ghi frame vào video đầu ra
            out.write(image)
            
            # Hiển thị frame trong thời gian thực (Tùy chọn)
            # cv2.imshow('Video Prediction', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        print(f"--- Xử lý hoàn tất. Video đã lưu tại {OUTPUT_VIDEO_PATH} ---")
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Đảm bảo cấu hình GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    predict_on_video()

    