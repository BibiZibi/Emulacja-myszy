import cv2
import numpy as np
import pyautogui
import time

# ----------------------------------------
# KONFIGURACJA
# ----------------------------------------
kamera = cv2.VideoCapture(0)
kaskada_twarzy = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Rozmiar obrazu
h, w = 480, 640
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# Prawa środkowa strefa (ROI) do kliknięcia
roi_lewy = int(w * 0.65)
roi_prawy = w
roi_gora = int(h * 0.3)
roi_dol = int(h * 0.7)

# Współczynnik czułości ruchu kursora
czułość = 2.5

# Czas ostatniego kliknięcia (sekundy)
ostatnie_klik = 0
czas_odstep = 1.0

# Pobierz rozdzielczość ekranu
screen_w, screen_h = pyautogui.size()

# ----------------------------------------
# PĘTLA GŁÓWNA
# ----------------------------------------
while True:
    ret, klatka = kamera.read()
    if not ret:
        break

    klatka = cv2.resize(klatka, (w, h))
    szara = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(szara, (7,7), 0)

    # -----------------------------
    # WYKRYWANIE TWARZY
    # -----------------------------
    twarze = kaskada_twarzy.detectMultiScale(blur, 1.1, 5, minSize=(60,60))
    if len(twarze) > 0:
        x, y, szer, wys = max(twarze, key=lambda rect: rect[2]*rect[3])
        cx = x + szer//2
        cy = y + wys//2
        cv2.rectangle(klatka, (x,y), (x+szer, y+wys), (0,255,0), 2)
        cv2.circle(klatka, (cx, cy), 5, (0,0,255), -1)

        # mapowanie środka twarzy na ekran
        cursor_x = np.interp(cx, [0,w], [0,screen_w])
        cursor_y = np.interp(cy, [0,h], [0,screen_h])
        pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)

    # -----------------------------
    # WYKRYWANIE RĘKI PO KSZTAŁCIE
    # -----------------------------
    # prosta segmentacja ruchomego obiektu (ręki)
    _, maska = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
    maska = cv2.medianBlur(maska, 5)

    kontury, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for kontur in kontury:
        pole = cv2.contourArea(kontur)
        if pole > 1500:  # minimalny rozmiar ręki / obiektu
            x, y, szer, wys = cv2.boundingRect(kontur)
            srodek_x = x + szer//2
            srodek_y = y + wys//2

            # sprawdzenie czy w ROI
            if roi_lewy < srodek_x < roi_prawy and roi_gora < srodek_y < roi_dol:
                cv2.rectangle(klatka, (x,y), (x+szer,y+wys), (0,0,255), 2)
                cv2.putText(klatka, "OBIEKT W ROI", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

                # jednokrotne kliknięcie
                obecny_czas = time.time()
                if obecny_czas - ostatnie_klik > czas_odstep:
                    pyautogui.click()
                    ostatnie_klik = obecny_czas
                break  # klikamy tylko raz na klatkę

    # -----------------------------
    # WYŚWIETLANIE
    # -----------------------------
    cv2.rectangle(klatka, (roi_lewy, roi_gora), (roi_prawy, roi_dol), (255,0,0),2)
    cv2.imshow("Sterowanie kursorem i klik", klatka)

    klawisz = cv2.waitKey(1) & 0xFF
    if klawisz == 27 or klawisz == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
