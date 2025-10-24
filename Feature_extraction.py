import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# Chọn nguồn video: nhập đường dẫn file hoặc nhấn Enter để dùng webcam
video_path = input("Nhập đường dẫn video (hoặc nhấn Enter để dùng webcam): ").strip()
if video_path:
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Không mở được file video: {video_path}")
    exit(1)
else:
  cap = cv2.VideoCapture(0)

def extract_keypoints(results) -> np.ndarray:
  """Convert MediaPipe Holistic results to a 1D numpy array.

  Layout used (concatenated in this order):
  - face: 468 landmarks * (x,y,z) => 468*3
  - pose: 33 landmarks * (x,y,z,visibility) => 33*4
  - left_hand: 21 landmarks * (x,y,z) => 21*3
  - right_hand: 21 landmarks * (x,y,z) => 21*3

  If a landmark set is missing (None), it's replaced with zeros of the expected size.
  Returns a flat float32 numpy array.
  """
  # Face
  if results.face_landmarks:
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark], dtype=np.float32).flatten()
  else:
    face = np.zeros(468 * 3, dtype=np.float32)

  # Pose (x, y, z, visibility)
  if results.pose_landmarks:
    pose = np.array([[lmk.x, lmk.y, lmk.z, getattr(lmk, 'visibility', 0.0)] for lmk in results.pose_landmarks.landmark], dtype=np.float32).flatten()
  else:
    pose = np.zeros(33 * 4, dtype=np.float32)

  # Left hand
  if results.left_hand_landmarks:
    lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
  else:
    lh = np.zeros(21 * 3, dtype=np.float32)

  # Right hand
  if results.right_hand_landmarks:
    rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
  else:
    rh = np.zeros(21 * 3, dtype=np.float32)

  return np.concatenate([face, pose, lh, rh])


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  sequence = []  # list of per-frame keypoint arrays
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Extract keypoints and append to sequence
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    
    if len(sequence) > 0:
        arr = np.stack(sequence)
        np.save('keypoints.npy', arr)
        print(f"Saved {arr.shape} to keypoints.npy")
    else:
        print("No keypoints collected yet.")
  # end while
  # When loop exits, save collected sequence (if any)
  if len(sequence) > 0:
    arr = np.stack(sequence)
    np.save('keypoints.npy', arr)
    print(f"Saved {arr.shape} to keypoints.npy")
  else:
    print("No keypoints were collected.")

cap.release()