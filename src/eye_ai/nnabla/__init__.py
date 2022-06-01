from os.path import isfile

from numpy import reshape
from nnabla import clear_parameters, auto_forward
from nnabla.utils import image_utils
from nnabla.utils.load import load
from nnabla.utils.cli.utility import let_data_to_variable

from eye_ai.images import adjust_shape
from eye_ai.explain.gradcam import generate_gradcam


class Nnabla:
    can_get_grad = True

    def __init__(self,
                 file, do_backpropagation=False):
        self.do_backpropagation = do_backpropagation
        self.is_ready = False

        clear_parameters()

        if not isfile(file):
            raise RuntimeError()

        info = load([file], prepare_data_iterator=False, batch_size=1)

        # noinspection PyUnresolvedReferences
        self.global_config = info.global_config
        # noinspection PyUnresolvedReferences
        self.executors = info.executors.values()

        if len(self.executors) < 1:
            raise RuntimeError()

        executor = self.networks = []
        # noinspection PyUnresolvedReferences
        if executor.network.name in info.networks.keys():
            # noinspection PyUnresolvedReferences
            self.networks.append(info.networks[executor.network.name])
        else:
            raise RuntimeError()

        # noinspection PyUnresolvedReferences
        if len(executor.dataset_assign.items()) > 1:
            raise RuntimeError()

    @property
    def executor(self):
        return list(self.executors)[0]

    @property
    def input_variable(self):
        return list(self.executor.dataset_assign.items())[0]

    @property
    def output_variable(self):
        return list(self.executor.dataset_assign.items())[0]

    @property
    def genre(self):
        return ''

    def inspect(self, image):
        if not self.is_ready:
            raise RuntimeError()

        if self.do_backpropagation:
            for v in self.executor.network.variables.values():
                v.variable_instance.need_grad = True
                v.variable_instance.grad.zero()

        input_variable, data_name = self.input_variable
        input_shape = input_variable.variable_instance.d.shape
        output_variable = self.output_variable
        image = self.preprocess(image, input_shape)

        let_data_to_variable(
            input_variable.variable_instance,
            reshape(image, input_shape),
            data_name=data_name,
            variable_name=input_variable.name
        )

        if self.do_backpropagation:
            input_variable.variable_instance.need_grad = True

        for v, generator in self.executor.generator_assign.items():
            v.variable_instance.d = generator(v.shape)

        return

    def preprocess(self, image, input_shape):
        _, color, h, w = input_shape
        image = image_utils.imresize(image, (w, h))
        image = adjust_shape(image, input_shape)
        if not self.executor.no_image_normalization:
            image = image / 255.0
        return image

    def backward(self, image, class_index):
        if not self.do_backpropagation:
            raise RuntimeError()

        with auto_forward():
            selected = self.output_variable.variable_instance[:, class_index]

        layers = []
        for k, v in self.executor.network.variables.items():
            v.variable_instance.g = 0
            layers.append((k, v))
        selected.backward()
        return generate_gradcam(image, layers)
