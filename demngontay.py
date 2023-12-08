import cv2
import time
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp.solutions.hands.Hands(model_complexity = 0, min_detection_confidence = 0.5, min_tracking_confidence =0.5)

cap = cv2.VideoCapture(0)
pTime = time.time()
while True:
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'fps: {int(fps)}', (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            handLandmarks = []

            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])

            #thumb
            if handLabel == 'Left' and handLandmarks[4][0] > handLandmarks[3][0]:
                finger_count += 1
            elif handLabel == 'Right' and handLandmarks[4][0] < handLandmarks[3][0]:
                finger_count +=1

            #other fingers
            if handLandmarks[8][1] < handLandmarks[6][1]:
                finger_count +=1
            if handLandmarks[12][1] < handLandmarks[10][1]:
                finger_count +=1
            if handLandmarks[16][1] < handLandmarks[14][1]:
                finger_count +=1
            if handLandmarks[20][1] < handLandmarks[18][1]:
                finger_count +=1
                
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            
    cv2.putText(frame, str(finger_count), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    cv2.imshow('cam day', frame)
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()