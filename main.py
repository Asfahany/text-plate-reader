import cv2
import easyocr

# Inisialisasi Reader EasyOCR untuk Bahasa Inggris/Angka
print("Memuat model OCR...")
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke Grayscale untuk pemrosesan teks yang lebih jernih
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi teks menggunakan EasyOCR
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        # Hanya tampilkan jika tingkat kepastian (probability) > 40%
        if prob > 0.4:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # Buat kotak hijau di sekeliling teks
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
            # Tampilkan teks hasil pembacaan di atas kotak
            cv2.putText(frame, f"{text} ({prob:.2f})", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Real-time OCR Reader", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
