import glfw

class FPSCounter:
    """
    calculates frames per second and seconds per frame

    """
    def __init__(self, frame_interval=100.0, spf_interval=10.0):
        self.frame_interval = frame_interval
        self.refresh_time = True
        self.st = glfw.get_time()
        self.st_spf = glfw.get_time()
        self.frame_count = 0
        self.seconds_per_frame = 1.0/260.0
        self.spf_interval = spf_interval

    def update(self):
        if self.refresh_time:
            self.st = glfw.get_time()
            self.refresh_time = False
        self.frame_count += 1
        if self.frame_count == self.frame_interval:
            self.refresh_time = True
            et = glfw.get_time()
            fps = self.frame_interval / (et - self.st)
            print(fps, " FPS")
            self.frame_count = 0
            self.fps = fps
        return self.update_spf()

    def update_spf(self):
        if self.frame_count % self.spf_interval == 0:
            et = glfw.get_time()
            self.seconds_per_frame = (et - self.st_spf) / (self.spf_interval)
            self.st_spf = glfw.get_time()
        return self.seconds_per_frame
