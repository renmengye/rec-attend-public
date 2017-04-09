class StepCounter(object):

    def __init__(self, step=0):
        self._step = step

    def increment(self):
        self._step += 1
        pass

    def set(self, value):
        self._step = step

    def get(self):
        return self._step

    def reset(self):
        self._step = 0
