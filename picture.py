import cv2
import os

# ---------------- CONFIG ----------------
video_folder = "videos"          # folder berisi banyak video
output_folder = "dataset_images" # folder output dataset final
interval = 5                     # ambil 1 frame tiap 5 frame
blur_threshold = 80              # threshold blur (50–120 bagus)
# -----------------------------------------

os.makedirs(output_folder, exist_ok=True)

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

saved_total = 0

# Loop semua file di folder "videos"
for filename in os.listdir(video_folder):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_path = os.path.join(video_folder, filename)
    print(f"\nMemproses: {filename}")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % interval == 0:
            blurry, score = is_blurry(frame, blur_threshold)

            if not blurry:
                save_path = os.path.join(output_folder, f"{filename}_frame_{saved_total}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"  Saved: {os.path.basename(save_path)} | Sharpness: {score:.2f}")
                saved_total += 1
                saved_count += 1
            else:
                print(f"  Skip blur (score {score:.2f})")

        frame_count += 1

    cap.release()
    print(f"Selesai video {filename}: tersimpan {saved_count} gambar")

print("\n======================================")
print(f"TOTAL GAMBAR TERSIMPAN: {saved_total}")
print("Dataset Siap Upload ke Roboflow!")
print("======================================")