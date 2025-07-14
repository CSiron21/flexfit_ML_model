import traceback
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    import pandas as pd
    import os
    # Load MoveNet Thunder model
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet = model.signatures['serving_default']
    # Function to preprocess image and run model
    def detect_keypoints(img):
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
        img = tf.cast(img, dtype=tf.int32)
        outputs = movenet(img)
        keypoints = outputs['output_0'].numpy()[0][0]
        return keypoints
    # Skeleton drawing connections (COCO format)
    SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
        (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    # Ensure output directory exists
    os.makedirs('train_data', exist_ok=True)
    # Ensure images output directory exists
    images_dir = os.path.join('train_data', 'images')
    os.makedirs(images_dir, exist_ok=True)
    # Process all videos in dataset directory
    video_dir = 'dataset'
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} video(s) in '{video_dir}': {video_files}")
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            continue
        frame_num = 0
        data = []
        img_save_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or cannot read frame at frame {frame_num} in {video_file}")
                break
            if frame_num % 3 == 0:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keypoints = detect_keypoints(rgb)
                    data.append(keypoints.flatten())
                    # Draw skeleton on blank image
                    blank = np.zeros((256, 256), dtype=np.uint8)
                    points = keypoints[:, :2] * [256, 256]  # scale to image size
                    points = points.astype(int)
                    for x, y, conf in keypoints:
                        if conf > 0.2:
                            cv2.circle(blank, (int(x*256), int(y*256)), 2, 255, -1)
                    for i, j in SKELETON_EDGES:
                        if keypoints[i, 2] > 0.2 and keypoints[j, 2] > 0.2:
                            pt1 = (int(keypoints[i, 0]*256), int(keypoints[i, 1]*256))
                            pt2 = (int(keypoints[j, 0]*256), int(keypoints[j, 1]*256))
                            cv2.line(blank, pt1, pt2, 255, 1)
                    # Resize to 128x128
                    skeleton_img = cv2.resize(blank, (128, 128), interpolation=cv2.INTER_AREA)
                    # Save as PNG
                    img_filename = f"{os.path.splitext(video_file)[0]}_f{img_save_idx:04d}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    cv2.imwrite(img_path, skeleton_img)
                    print(f"[OK] Saved skeleton image: {img_path}")
                    img_save_idx += 1
                    print(f"Processed frame {frame_num} in {video_file}")
                except Exception as e:
                    print(f"[ERROR] Exception at frame {frame_num} in {video_file}: {e}")
            frame_num += 1
        cap.release()
        if len(data) == 0:
            print(f"[WARNING] No frames processed for {video_file}. No CSV will be saved.")
            continue
        columns = [f"{coord}_{i}" for i in range(17) for coord in ["x", "y", "score"]]
        df = pd.DataFrame(data, columns=columns)
        out_csv = os.path.join('train_data', f"{os.path.splitext(video_file)[0]}_keypoints.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Keypoints saved to {out_csv}")
except Exception as e:
    print("[FATAL ERROR] An exception occurred during script execution:")
    traceback.print_exc()
