import importlib
import numpy as np
import tensorflow as tf

# tflite = importlib.import_module('tensorflow.compat.v1').lite
try:
    tflite = importlib.import_module('tensorflow.compat.v1').lite
except ModuleNotFoundError:
    try:
        tflite = importlib.import_module('tflite_runtime.interpreter')
    except ModuleNotFoundError:
        print('Either Tensorflow or tflite_runtime must be installed.')
        raise


def get_model_detail(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    return (interpreter.get_input_details(), interpreter.get_output_details())


class Model:
    MODEL_SAMPLE_RATE = 16000
    MODEL_WINDOW_LENGTH = 2048

    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        # Needed before allocation
        self.interpreter.allocate_tensors()

        self._input_details, self._output_details = get_model_detail(model_path)

        self._output_index = {
            detail['name']: detail['index'] for detail in self._output_details
        }

        self._input_wav_length = self._input_details[0]['shape'][0]
        self._output_roll_length = self._output_details[0]['shape'][1]

        assert (self._input_wav_length -
                Model.MODEL_WINDOW_LENGTH) % (self._output_roll_length - 1) == 0

        self._hop_size = (self._input_wav_length - Model.MODEL_WINDOW_LENGTH) // (
                self._output_roll_length - 1)

        self._timestep = float(self._hop_size) / Model.MODEL_SAMPLE_RATE

        for i, v in self._output_index.items():
            print(i, v)

        print("hop_size = ", self._hop_size, "output_roll = ", self._output_roll_length)

    @property
    def sample_rate(self):
        return Model.MODEL_SAMPLE_RATE

    @property
    def window_length(self):
        return Model.MODEL_WINDOW_LENGTH

    @property
    def timestep(self):
        """Returns the clock time in ms"""
        return int(1000 * self._timestep)

    @property
    def input_wav_length(self):
        return self._input_wav_length

    @property
    def hop_size(self):
        return self._hop_size


if __name__ == '__main__':
    Model("./pretrained/onsets_frames_wavinput_uni.tflite")
