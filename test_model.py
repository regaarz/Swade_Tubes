from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak ditemukan!")
    exit()

threshold = 0.9  # nilai threshold agar tidak asal prediksi

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Predict
    results = model.predict(frame)[0]

    # Predict result
    top_class = results.probs.top1
    label = results.names[top_class]
    conf = float(results.probs.top1conf)

    # Gunakan threshold
    if conf < threshold:
        label_show = "Tidak ada objek"
        text = f"{label_show}"
    else:
        label_show = label
        text = f"{label_show}: {conf:.2f}"

    # Tampilkan teks
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Classification - Webcam (Threshold)", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
