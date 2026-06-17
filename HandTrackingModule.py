import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=0,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Custom overlay style: bright magenta/pink dots, bright green lines.
        # BGR order, so magenta = (255, 0, 255), green = (0, 255, 0).
        self.lmStyle = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4)
        self.connStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)

        # Holds the latest detection so findPosition can run safely even if it
        # gets called before findHands on a given frame.
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.lmStyle,
                        self.connStyle,
                    )
        return img

    def findPosition(self, img, handNo=0, draw=False):
        lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        return lmList

    def get_raw_landmarks(self, handNo=0):
        """Return the raw MediaPipe hand_landmarks object for the given hand,
        or None if that hand isn't present.

        Unlike findPosition (which gives pixel x, y), this exposes the
        normalized lm.x, lm.y, lm.z needed for machine learning features.
        Call findHands() first so self.results is populated.
        """
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                return self.results.multi_hand_landmarks[handNo]
        return None


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()