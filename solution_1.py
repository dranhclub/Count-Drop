'''
Solution 1: Tính toán và dự đoán khoảng rơi theo trục y
63 / 63
'''
import numpy as np
import cv2 as cv
import math
from particle import Particle

if __name__ == "__main__":
    cap = cv.VideoCapture("Basler_acA1300-30gc__21472292__20210305_164914214 (2).avi")
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bg = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    FRAME_WIDTH = int(cap.get(4))
    FRAME_HEIGHT = int(cap.get(3))

    y0 = 0.003
    H = 0.06  # 75mm
    ratio = FRAME_HEIGHT / H
    g = 9.8  # gravity
    offset_y = 0
    repeat = False
    FPS = cap.get(cv.CAP_PROP_FPS)

    print("y0=", y0)
    print("H=", H)
    print("ratio=", ratio)
    print("g=", g)
    print("offset_y=", offset_y)
    print("repeat=", repeat)
    print("FPS=", FPS)

    def next_y(y):
        y = y / ratio
        t1 = math.sqrt(2 * y / g)
        t2 = t1 + 1 / FPS
        epsilon = 0.8 / FPS
        y2_left = (g * y0 * t2 + 0.5 * g * (t2 - epsilon) ** 2) * ratio
        y2_right = (g * y0 * t2 + 0.5 * g * (t2 + epsilon) ** 2) * ratio
        return int(y2_left), int(y2_right)


    frame_count = 0
    # prev_centroids = []
    count = 0
    particles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            if repeat:
                print('repeat')
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                count = 0
                prev_centroids = []
                continue
            else:
                print("Count: ", count)
                break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        diff = cv.absdiff(frame, bg)
        diff = cv.GaussianBlur(diff, (5, 5), 0)
        diff = cv.medianBlur(diff, 5)
        ret, thresh = cv.threshold(diff, 15, 255, cv.THRESH_BINARY)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
        # cv.imshow("thresh", thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        centroids = []
        # mask = [False for i in range(0, len(prev_centroids))]

        for cnt in contours:
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

        mask = [False for i in range(0, len(contours))]
        for particle in particles:
            if particle.onscreen:
                particle.track(contours, centroids, next_y, mask)
        for i, m in enumerate(mask):
            if not m:
                count += 1
                new_particle = Particle(contours[i], centroids[i], count)
                particles.append(new_particle)

        # drawing
        for particle in particles:
            if not particle.onscreen:
                continue

            # draw centroid
            cv.circle(frame, particle.centroid, 2, (0, 0, 255), -1)

            # draw bounding rect
            x, y, w, h = cv.boundingRect(particle.contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw next_y line
            cx, cy = particle.centroid
            next_y_left, next_y_right = next_y(cy)
            cv.rectangle(frame, (0, next_y_left), (FRAME_WIDTH, next_y_right), (255, 255, 0))

            # draw number of the particle
            cv.putText(frame, str(particle.number), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

            # particle.update()

        cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
        # print(particles)

        cv.putText(frame, "frame: " + str(frame_count), (0, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        cv.putText(frame, "count: " + str(count), (0, FRAME_HEIGHT - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        # offset line
        cv.line(frame, (0, offset_y), (FRAME_WIDTH, offset_y), (255, 255, 255), 2)
        cv.line(frame, (0, FRAME_HEIGHT - offset_y), (FRAME_WIDTH, FRAME_HEIGHT - offset_y), (255, 255, 255), 2)
        cv.imshow("contour", frame)

        frame_count += 1
        cv.waitKey(0)

    cv.destroyAllWindows()
