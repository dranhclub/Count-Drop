import cv2 as cv
from pypylon import pylon
import math
from particle import Particle
from datetime import datetime
import re
from video_input import BaslerCameraVideoInput, RawVideoInput

bg = None
FRAME_WIDTH = 0
FRAME_HEIGHT = 0
FPS = 0

y0 = 0.007  # initial height
H = 0.06  # real height of working area
ratio = 0
g = 9.8  # gravity
offset_y = 0
record = False

video_writer = cv.VideoWriter()


def init(background, width, height, fps):
    global bg, FRAME_HEIGHT, FRAME_WIDTH, FPS, ratio, record, video_writer
    bg = background
    FRAME_WIDTH = width
    FRAME_HEIGHT = height
    FPS = fps
    ratio = FRAME_HEIGHT / H

    if record:
        d = str(datetime.now())
        d = re.sub(":", "-", d)
        filename = "video_record/" + d + ".avi"
        print(filename)
        codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv.VideoWriter(filename, codec, 10, (FRAME_WIDTH, FRAME_HEIGHT))

    print("Initialize: ")
    print("WIDTH={}, HEIGHT={}".format(FRAME_WIDTH, FRAME_HEIGHT))
    print("y0=", y0)
    print("H=", H)
    print("ratio=", ratio)
    print("g=", g)
    print("offset_y=", offset_y)
    print("FPS=", FPS)
    print("record=", record)

    cv.imshow("Background captured", bg)


def next_y(y1):
    y1 = y1 / ratio
    v0 = math.sqrt(2 * y0 * g)
    t1 = (-v0 + math.sqrt(v0 * v0 + 2 * g * y1)) / g
    t2 = t1 + 1 / FPS
    epsilon = 0.3 / FPS
    t2_left = t2 - epsilon
    t2_right = t2 + epsilon
    y2_left = (v0 * t2_left + 0.5 * g * t2_left ** 2) * ratio
    y2_right = (v0 * t2_right + 0.5 * g * t2_right ** 2) * ratio
    return int(y2_left), int(y2_right)


def thresh_binary(frame, background):
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(frame, background)
    diff = cv.GaussianBlur(diff, (5, 5), 0)
    diff = cv.medianBlur(diff, 5)
    ret, thresh = cv.threshold(diff, 15, 255, cv.THRESH_BINARY)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
    return thresh


def get_centroid(contour):
    m = cv.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


def update_particles(contours, old_particles):
    mask = [False for i in range(0, len(contours))]
    for particle in old_particles:
        if particle.onscreen:
            particle.track(contours, next_y, mask)
    for i, m in enumerate(mask):
        if not m:
            new_particle = Particle(contours[i])
            old_particles.append(new_particle)


def draw_info(frame, particles, frame_count):
    # particles info
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

    # frame number
    cv.putText(frame, "frame: " + str(frame_count), (0, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

    # count number
    cv.putText(frame, "count: " + str(len(particles)), (0, FRAME_HEIGHT - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

    # offset line
    cv.line(frame, (0, offset_y), (FRAME_WIDTH, offset_y), (255, 255, 255), 2)
    cv.line(frame, (0, FRAME_HEIGHT - offset_y), (FRAME_WIDTH, FRAME_HEIGHT - offset_y), (255, 255, 255), 2)

    return frame


def start_count(video_source):
    frame_count = 0
    particles = []
    for frame in video_source.frames():
        frame_count += 1
        thresh = thresh_binary(frame, bg)
        cv.imshow("thresh", thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
        update_particles(contours, particles)

        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        frame = draw_info(frame, particles, frame_count)
        cv.imshow("Counting", frame)
        if record:
            global video_writer
            video_writer.write(frame)

        if cv.waitKey(1) == 27:
            break

    video_writer.release()

    return len(particles)


if __name__ == "__main__":
    # video_source = BaslerCameraVideoInput(cv.ROTATE_90_COUNTERCLOCKWISE)
    video_source = RawVideoInput("video_input/Basler_acA1300-30gc__21472292__20210305_164914214 (2).avi",
                                 cv.ROTATE_90_COUNTERCLOCKWISE)

    # Wait for init
    for frame in video_source.frames():
        cv.imshow("Counting", frame)
        if cv.waitKey(0) == ord('g'):
            init(frame, video_source.width, video_source.height, video_source.get_fps())
            break

    # Start counting
    count = start_count(video_source)
    print("***************************************")
    print("Count = ", count)

    video_source.release()
    cv.destroyAllWindows()

