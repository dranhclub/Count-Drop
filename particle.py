import cv2 as cv


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

    def track(self, contours, centroids, next_y, mask):
        left, right = next_y(self.centroid[1])
        for i, (x, y) in enumerate(centroids):
            if mask[i]:
                continue
            if left <= y <= right:
                mask[i] = True
                self.centroid = (x, y)
                self.contour = contours[i]
                return
        else:
            self.onscreen = False

    def __repr__(self):
        x, y = self.centroid
        return "<{}, ({},{}), {}>".format(self.number, x, y, self.onscreen)
