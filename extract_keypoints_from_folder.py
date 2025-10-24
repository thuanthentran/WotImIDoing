import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
import time

# --- CẤU HÌNH CHUNG ---
VIDEO_ROOT = 'Videos'       # Thư mục chứa các folder nhãn (label)
KEYPOINTS_ROOT = 'Keypoints' # Thư mục lưu trữ file .npy đầu ra
MAX_FRAMES = 120            # Độ dài chuỗi (2s * 60 FPS)
# Kích thước đặc trưng: (468 Face + 2*21 Hands + 33 Pose) * 3 + (2*21 Hands) * 3 = 1755
NUM_FEATURES = 1755 
POSE_NOSE_INDEX = 0 

# --- THIẾT LẬP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic

# --------------------------------------------------------------------------------------
# HÀM TRÍCH XUẤT VÀ CHUẨN HÓA KEYPOINTS (Tối ưu: Chuẩn hóa Kép)
# --------------------------------------------------------------------------------------

def extract_and_normalize_keypoints(results):
    """
    Trích xuất và Chuẩn hóa Keypoints: Toàn cục (so với Mũi) và Cục bộ (so với Cổ tay).
    """
    
    # Lấy tọa độ thô của Pose, Hands, Face (x, y, z)
    pose_raw = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else None
    face_raw = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else None
    lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None
    rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else None

    # --- 1. CHUẨN HÓA TOÀN CỤC (GLOBAL NORMALIZATION) ---
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
    
    # --- 2. CHUẨN HÓA CỤC BỘ (LOCAL NORMALIZATION) - Hình dáng bàn tay ---
    hand_shape_features = []
    
    # Tay Trái
    if lh_raw is not None:
        lh_wrist = lh_raw[0] 
        lh_local_norm = (lh_raw - lh_wrist).flatten() 
        hand_shape_features.append(lh_local_norm)
    else:
        hand_shape_features.append(np.zeros(21 * 3))

    # Tay Phải
    if rh_raw is not None:
        rh_wrist = rh_raw[0] 
        rh_local_norm = (rh_raw - rh_wrist).flatten()
        hand_shape_features.append(rh_local_norm)
    else:
        hand_shape_features.append(np.zeros(21 * 3))

    # --- 3. KẾT HỢP TẤT CẢ ---
    final_keypoints = np.concatenate([
        face_global_norm, lh_global_norm, rh_global_norm, pose_global_norm,
        hand_shape_features[0], hand_shape_features[1]  
    ])
    
    if final_keypoints.size != NUM_FEATURES:
         return np.zeros(NUM_FEATURES) 
         
    return final_keypoints

# --------------------------------------------------------------------------------------
# HÀM XỬ LÝ ĐƠN VIDEO (Đã sửa lỗi tham chiếu)
# --------------------------------------------------------------------------------------

def process_single_video(task_info):
    """
    Xử lý một video duy nhất trong một tiến trình độc lập.
    Args: (video_file, action_label, output_dir)
    """
    video_file, action, output_action_dir = task_info
    
    # Khởi tạo mô hình MediaPipe Holistic trong MỖI tiến trình
    with mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as holistic:
        
        sequence = []
        cap = cv2.VideoCapture(video_file)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            results = holistic.process(image)
            
            # Lỗi được sửa tại đây: hàm đã được định nghĩa và có thể gọi được
            keypoints = extract_and_normalize_keypoints(results) 
            sequence.append(keypoints)
            
        cap.release()

        if sequence:
            video_filename = os.path.basename(video_file).replace('.mp4', '')
            output_filepath = os.path.join(output_action_dir, f'{video_filename}.npy')
            np.save(output_filepath, np.array(sequence, dtype=np.float32))
            print(f"[SUCCESS] {action}/{video_filename}: {len(sequence)} khung hình.")
        else:
            print(f"[FAIL] {action}/{os.path.basename(video_file)}: Không trích xuất được keypoints.")

# --------------------------------------------------------------------------------------
# HÀM CHÍNH SỬ DỤNG POOL MULTIPROCESSING
# --------------------------------------------------------------------------------------

def process_video_data_multiprocessing():
    """Lập danh sách tất cả các video và phân phối cho các tiến trình."""
    print("--- Bắt đầu GIAI ĐOẠN 1: Trích xuất Keypoints (ĐA TIẾN TRÌNH) ---")
    start_time = time.time()
    os.makedirs(KEYPOINTS_ROOT, exist_ok=True)
    
    # 1. Lập danh sách TẤT CẢ các tác vụ (tasks)
    all_tasks = []
    actions = sorted(os.listdir(VIDEO_ROOT))
    if not actions:
        print(f"LỖI: Không tìm thấy thư mục nhãn nào trong {VIDEO_ROOT}. Hãy kiểm tra cấu trúc.")
        return
        
    for action in actions:
        action_dir = os.path.join(VIDEO_ROOT, action)
        if os.path.isdir(action_dir):
            output_action_dir = os.path.join(KEYPOINTS_ROOT, action)
            os.makedirs(output_action_dir, exist_ok=True)
            
            video_files = sorted(glob.glob(os.path.join(action_dir, '*.mp4')))
            
            for video_file in video_files:
                video_filename = os.path.basename(video_file).replace('.mp4', '')
                output_filepath = os.path.join(output_action_dir, f'{video_filename}.npy')
                
                # Bỏ qua video đã được trích xuất
                if not os.path.exists(output_filepath):
                    all_tasks.append((video_file, action, output_action_dir))
    
    total_videos = len(all_tasks)
    if total_videos == 0:
        print("Không có video mới để xử lý. Kiểm tra thư mục Keypoints.")
        return

    # 2. Định cấu hình số lượng tiến trình
    num_processes = cpu_count() - 1 
    if num_processes < 1: num_processes = 1
    
    print(f"Tổng số video cần xử lý: {total_videos}")
    print(f"Sử dụng {num_processes} tiến trình.")

    # 3. Chạy đa tiến trình
    with Pool(num_processes) as pool:
        pool.map(process_single_video, all_tasks)

    end_time = time.time()
    print("\n--- GIAI ĐOẠN 1 HOÀN TẤT. ---")
    print(f"Thời gian xử lý: {end_time - start_time:.2f} giây")


if __name__ == '__main__':
    # BẮT BUỘC: Đa tiến trình trên Windows phải chạy trong khối này.
    process_video_data_multiprocessing()