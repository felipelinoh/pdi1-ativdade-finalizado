import cv2
import numpy as np

cap = cv2.VideoCapture("q1/q1A.mp4")
display_width, display_height = 1024, 576

def detectar_formas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red, upper_red = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_blue, upper_blue = np.array([100, 150, 70]), np.array([140, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    masks = [(red_mask, (0, 0, 255)), (blue_mask, (None))]
    
    detected_shapes = []

    for mask, color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                detected_shapes.append((x, y, w, h, area, (0, 255, 0)))
                if color is not None:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 8)
                
    return frame, detected_shapes

def detectar_colisao(shapes, frame):
    for i, (x1, y1, w1, h1, _, _) in enumerate(shapes):
        for j, (x2, y2, w2, h2, _, _) in enumerate(shapes):
            if i != j:
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2):
                    cv2.putText(frame, "COLISÃO DETECTADA", (1010, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                    
def detectar_ultrapassagem(shapes, frame):
    for i, (x1, y1, w1, h1, _, _) in enumerate(shapes):
        for j, (x2, y2, w2, h2, _, _) in enumerate(shapes):
            if i != j:
                if (x1 > x2 and x1 + w1 < x2 + w2 and
                    y1 > y2 and y1 + h1 < y2 + h2):
                    cv2.putText(frame, "ULTRAPASSAGEM DETECTADA", (1010, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    
while True:
    ret, frame = cap.read()

    if not ret:

        break

    frame, shapes = detectar_formas(frame)
    detectar_colisao(shapes, frame)
    detectar_ultrapassagem(shapes, frame)
    resized_frame = cv2.resize(frame, (display_width, display_height))

    
    # Exibe resultado
    cv2.imshow("Feed", resized_frame)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()

cv2.destroyAllWindows()
