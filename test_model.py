from ultralytics import YOLO
import cv2

# Load model (ganti path jika perlu)
model = YOLO("best.pt")

# Buka kamera (0 = webcam utama)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak ditemukan!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Predict dari frame
    results = model.predict(frame)[0]

    # Ambil prediksi top-1
    label = results.names[results.probs.top1]
    conf = float(results.probs.top1conf)

    # Tampilkan teks di frame
    text = f"{label}: {conf:.2f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Classification - Webcam", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
