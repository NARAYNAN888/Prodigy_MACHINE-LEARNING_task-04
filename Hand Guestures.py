import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Detect hands in the frame
    hands, _ = detector.findHands(frame)

    if hands:
        # Get information about each hand
        for hand in hands:
            handType = hand['type']  # 'Left' or 'Right'
            fingers = detector.fingersUp(hand)

            # Print the hand type and finger count
            cv2.putText(frame, f"{handType} Hand - Fingers: {sum(fingers)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
