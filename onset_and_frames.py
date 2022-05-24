import multiprocessing
import threading
from typing import List

from absl import app
from absl import flags
import attr

from magenta.models.onsets_frames_transcription.realtime import audio_recorder
from magenta.models.onsets_frames_transcription.realtime import tflite_model
import numpy as np

flags.DEFINE_string("model_path", "./pretrained/onsets_frames_wavinput_no_offset_uni.tflite")
flags.DEFINE_string('mic', None, 'Optional: Input source microphone ID.')
flags.DEFINE_float('mic_amplify', 30.0, 'Multiply raw audio mic input')
FLAGS = flags.FLAGS


def results_collector():
    pass


class OnsetTask:
    def __init__(self):
        pass


@attr.s
class AudioChunk(object):
    serial = attr.ib()
    samples = attr.ib(repr=lambda w: '{} {}'.format(w.shape, w.dtype))


class AudioQueue:
    """
    Audio Queue
    """

    def __init__(self, callback, audio_device_index, sample_rate_hz, model_sample_rate, frame_length, overlap):
        pass


class OnsetAndFrame:
    def __init__(self):
        self.results = multiprocessing.Queue()
        self.results_thread = threading.Thread(target=results_collector, args=(self.results,))

        pass

    def _set_model(self):
        self.model = tflite_model.Model(FLAGS.model_path)
        pass

    def start(self):
        pass

    def stop(self):
        pass


class TFLiteWorker(multiprocessing.Process):
    """Process for excuting TFLite inference"""

    def __init__(self, model_path, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._model = None

    def setup(self):
        if self._model is not None:
            return
        # model init if model is not exist
        self.model = tflite_model.Model(self._model_path)

    def run(self):
        self.setup()
        while True:
            # initialize job
            task = self._task_queue.get()
            if task is None:
                self._task_queue.task_done()
                return
            self._task_queue.task_done()
            self._result_queue.put(task)
