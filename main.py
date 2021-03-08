import cv2 as cv
from pypylon import pylon
import math
from particle import Particle

bg = None
FRAME_WIDTH = 0
FRAME_HEIGHT = 0
FPS = 0

y0 = 0.003
H = 0.06  # 75mm
ratio = 0
g = 9.8  # gravity
offset_y = 0
repeat = False


def init(grabResult, fps):
    global bg, FRAME_HEIGHT, FRAME_WIDTH, FPS, ratio
    bg = grabResult.Array
    bg = cv.rotate(bg, cv.ROTATE_90_COUNTERCLOCKWISE)
    FRAME_WIDTH = grabResult.Height
    FRAME_HEIGHT = grabResult.Width
    FPS = fps
    ratio = FRAME_HEIGHT / H

    print("Initialize: ")
    print("WIDTH={}, HEIGHT={}".format(FRAME_WIDTH, FRAME_HEIGHT))
    print("y0=", y0)
    print("H=", H)
    print("ratio=", ratio)
    print("g=", g)
    print("offset_y=", offset_y)
    print("repeat=", repeat)
    print("FPS=", FPS)

    cv.imshow("Background captured", bg)

camera = None

try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
except:
    exit("No device is available")


def wait_for_init():
    camera.StartGrabbing()
    timestamp = 0
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # calc fps
            t2 = grabResult.TimeStamp
            fps = 1000000000 / (t2 - timestamp) / 8
            timestamp = t2

            # get image
            img = grabResult.Array
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            cv.imshow("Counting", img)

            # wait key
            if cv.waitKey(1) == ord('g'):
                init(grabResult, fps)
                break

    grabResult.Release()
    camera.StopGrabbing()


def next_y(y):
    y = y / ratio
    t1 = -H + math.sqrt(H * H + 2 * y / g)
    t2 = t1 + 1 / FPS
    epsilon = 0.8 / FPS
    y2_left = (g * y0 * t2 + 0.5 * g * (t2 - epsilon) ** 2) * ratio
    y2_right = (g * y0 * t2 + 0.5 * g * (t2 + epsilon) ** 2) * ratio
    return int(y2_left), int(y2_right)


def thresh_binary(frame, background):
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    diff = cv.absdiff(frame, background)
    diff = cv.GaussianBlur(diff, (5, 5), 0)
    diff = cv.medianBlur(diff, 5)
    ret, thresh = cv.threshold(diff, 15, 255, cv.THRESH_BINARY)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
    return thresh


def get_centroid(contour):
    m = cv.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


def update_particles(contours, old_particles):
    centroids = [get_centroid(c) for c in contours]
    mask = [False for i in range(0, len(contours))]
    for particle in old_particles:
        if particle.onscreen:
            particle.track(contours, centroids, next_y, mask)
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


def start_count():
    frame_count = 0
    particles = []

    camera.StartGrabbing()
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            frame = grab_result.Array
            frame_count += 1
            thresh = thresh_binary(frame, bg)
            cv.imshow("thresh", thresh)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # cv.drawContours(frame, contours, -1, (0, 255, 0), 1)

            update_particles(contours, particles)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            frame = draw_info(frame, particles, frame_count)
            cv.imshow("Counting", frame)
        if cv.waitKey(1) == 27:
            break

    grab_result.Release()
    camera.StopGrabbing()

    return len(particles)


wait_for_init()
count = start_count()
camera.Close()
cv.destroyAllWindows()

print("***************************************")
print("Count = ", count)


