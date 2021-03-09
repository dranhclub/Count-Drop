import cv2 as cv


def get_centroid(contour):
    m = cv.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


class Particle:
    count = 0

    def __init__(self, contour):
        self.contour = contour
        m = cv.moments(contour)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        self.centroid = (cx, cy)
        self.onscreen = True
        Particle.count += 1
        self.number = Particle.count

    def track(self, contours, next_y, mask):
        left, right = next_y(self.centroid[1])
        for i, contour in enumerate(contours):
            cx, cy = get_centroid(contour)
            if mask[i]:
                continue
            if left <= cy <= right:
                mask[i] = True
                self.centroid = (cx, cy)
                self.contour = contours[i]
                return
        else:
            self.onscreen = False

    def __repr__(self):
        x, y = self.centroid
        return "<{}, ({},{}), {}>".format(self.number, x, y, self.onscreen)
