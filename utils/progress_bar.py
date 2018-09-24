import time
import datetime


class ProgressBar:
    def __init__(self, length, max_value):
        assert length > 0 and max_value > 0
        self.length, self.max_value, self.start = length, max_value, time.time()

    def update(self, value):
        assert 0 < value <= self.max_value
        delta = (time.time() - self.start) * (self.max_value - value) / value
        format_spec = [value / self.max_value,
                       value,
                       len(str(self.max_value)),
                       self.max_value,
                       len(str(self.max_value)),
                       '#' * int((self.length * value) / self.max_value),
                       self.length,
                       datetime.timedelta(seconds=int(delta))
                       if delta < 60 * 60 * 10 else '-:--:-']

        print('\r{:=5.0%} ({:={}}/{:={}}) [{:{}}] ETA: {}'.format(*format_spec), end='')
