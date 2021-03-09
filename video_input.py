from pypylon import pylon
import cv2 as cv


class IVideoInput:
    def frames(self):
        pass

    def get_fps(self):
        pass

    def release(self):
        pass


class RawVideoInput(IVideoInput):
    def __init__(self, filepath, rotate_code=-1):
        self.filepath = filepath
        self.rotate_code = rotate_code
        self.cap = cv.VideoCapture(filepath)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("cannot read video")

        self.__fps__ = self.cap.get(cv.CAP_PROP_FPS)
        if rotate_code == cv.ROTATE_90_COUNTERCLOCKWISE or rotate_code == cv.ROTATE_90_CLOCKWISE:
            self.width = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.height = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        else:
            self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    def frames(self):
        ret, frame = self.cap.read()
        while ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.rotate(frame, rotateCode=self.rotate_code)

            yield frame
            ret, frame = self.cap.read()

    def get_fps(self):
        return self.__fps__

    def release(self):
        self.cap.release()


class BaslerCameraVideoInput(IVideoInput):

    def __init__(self, rotate_code=-1):
        self.camera = None
        self.rotate_code = rotate_code
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
        except:
            exit("No device is available")
        self.camera.StartGrabbing()
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if rotate_code == cv.ROTATE_90_COUNTERCLOCKWISE or rotate_code == cv.ROTATE_90_CLOCKWISE:
            self.width = grab_result.Height
            self.height = grab_result.Width
        else:
            self.width = grab_result.Width
            self.height = grab_result.Height

    def frames(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = grab_result.Array
                if self.rotate_code != -1:
                    frame = cv.rotate(frame, self.rotate_code)
                yield frame

    def get_fps(self):
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        t1 = grab_result.TimeStamp
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        t2 = grab_result.TimeStamp
        return 1000000000 / (t2 - t1) / 8

    def release(self):
        self.camera.StopGrabbing()
        self.camera.Close()