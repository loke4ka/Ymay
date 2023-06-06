import cv2
from cvzone.HandTrackingModule import HandDetector


def detect_hands(frame):
    # Создайте экземпляр класса HandDetector для обнаружения рук
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Обработайте кадр для обнаружения рук
    hands, _ = detector.findHands(frame)

    # Получите координаты и контуры рук
    hand_data = []
    for hand in hands:
        hand_data.append({
            'hand_landmarks': hand['lmList'],  # Координаты ключевых точек рук
            'hand_bbox': hand['bbox'],  # Ограничивающий прямоугольник вокруг руки
            'hand_type': hand['type'],  # Тип руки (левая или правая)
        })

    return hand_data
