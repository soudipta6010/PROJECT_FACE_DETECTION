import cv2

a = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

b = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not b.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    c_rec, d_image = b.read()

    if not c_rec:
        print("Failed to grab frame")
        break
    # Convert to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, 1.3, 6)

    # Draw rectangles
    for x1, y1, w1, h1 in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    # Show the video feed
    cv2.imshow("Face Detection", d_image)

     # Press 'q' to quit
    h = cv2.waitKey(40) & 0xFF
    if h == ord('q'):
        break
    
b.release()
cv2.destroyAllWindows()
