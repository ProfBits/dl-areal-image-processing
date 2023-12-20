import json
import random
import sys
from enum import Enum
from typing import Callable, Optional, Sequence
import cv2
import parameterized.preprocessing as pre
import parameterized.detection as detect
import parameterized.postprocessing as post
import numpy as np


class _ParameterType(Enum):
    Continues = 0
    Categorical = 1


class __Parameter:
    def __init__(self, param_type: _ParameterType,
                 default: any,
                 values: Optional[Sequence[any]] = None,
                 limits: Optional[tuple[any, any]] = None,
                 is_int: bool = False):
        self.paramType = param_type
        self.default = default
        self.values = values
        self.limits = limits
        self.is_int = is_int


def __categorical_parameter(values: Sequence[any], default: any) -> __Parameter:
    return __Parameter(_ParameterType.Categorical, default, values=values)


def __bool_parameter(default: bool) -> __Parameter:
    return __Parameter(_ParameterType.Categorical, default, values=[True, False])


def __continues_parameter(limits: tuple[any, any], default: any, is_int: bool = False) -> __Parameter:
    return __Parameter(_ParameterType.Continues, default, limits=limits, is_int=is_int)


__morph_shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS]


__args: dict[str, __Parameter] = {
    "enable_shadow_reduction": __bool_parameter(True),
    "sr_convolve_window_size": __continues_parameter((1, 15), 5, True),
    "sr_num_thresholds": __continues_parameter((0, 7), 2, True),
    "sr_struc_elem_size": __continues_parameter((1, 15), 5, True),
    "sr_exponent": __continues_parameter((-4, 5), 1),
    "blurr_size": __continues_parameter((0, 7), 5, True),
    "blurr_sigma": __continues_parameter((0, 6), 0),
    "use_hsv": __bool_parameter(False),
    "hsv_limit0_H_MIN": __continues_parameter((0, 255), 64, True),
    "hsv_limit0_H_MAX": __continues_parameter((0, 255), 150, True),
    "hsv_limit0_S_MIN": __continues_parameter((0, 100), 0, True),
    "hsv_limit0_S_MAX": __continues_parameter((0, 100), 100, True),
    "hsv_limit0_V_MIN": __continues_parameter((0, 100), 20, True),
    "hsv_limit0_V_MAX": __continues_parameter((0, 100), 80, True),
    "hsv_limit1_H_MIN": __continues_parameter((0, 255), 40, True),
    "hsv_limit1_H_MAX": __continues_parameter((0, 255), 60, True),
    "hsv_limit1_S_MIN": __continues_parameter((0, 100), 30, True),
    "hsv_limit1_S_MAX": __continues_parameter((0, 100), 70, True),
    "hsv_limit1_V_MIN": __continues_parameter((0, 100), 20, True),
    "hsv_limit1_V_MAX": __continues_parameter((0, 100), 80, True),
    "rgb_red_weight": __continues_parameter((0, 1), 0.5),
    "rgb_rb_offset": __continues_parameter((-255, 255), 0),
    "rgb_green_weight": __continues_parameter((0, 2), 1),
    "rgb_sub_weight": __continues_parameter((0, 2), 1),
    "rgb_sub_offset": __continues_parameter((-255, 255), 0),
    "bin_threshold": __continues_parameter((0, 255), 127, True),
    "hist_slope": __continues_parameter((-127, 127), 4),
    "hist_offset": __continues_parameter((-127, 127), 0),
    "morph_shape_open": __categorical_parameter(__morph_shapes, cv2.MORPH_RECT),
    "morph_size_open_x": __continues_parameter((1, 9), 3, True),
    "morph_size_open_y": __continues_parameter((1, 9), 3, True),
    "morph_shape_close": __categorical_parameter(__morph_shapes, cv2.MORPH_RECT),
    "morph_size_close_x": __continues_parameter((1, 9), 3, True),
    "morph_size_close_y": __continues_parameter((1, 9), 3, True),
    "morph_open_first": __bool_parameter(False)
}


def __run(image: str | cv2.typing.MatLike, output: Optional[str], parameters: dict[str, any]) -> cv2.typing.MatLike:
    if parameters["enable_shadow_reduction"] is True:
        image = pre.remove_shadows(image,
                                   convolve_window_size=(parameters["sr_convolve_window_size"] * 2) + 1,
                                   num_thresholds=parameters["sr_num_thresholds"],
                                   struc_elem_size=parameters["sr_struc_elem_size"],
                                   exponent=parameters["sr_exponent"])

    image = pre.gaussian_blur(image,
                              size=(parameters["blurr_size"] * 2) + 1,
                              sigma=parameters["blurr_sigma"])

    if parameters["use_hsv"] is True:
        image = detect.hsv_detection(image,
                                     limits=[
                                         {
                                             detect.Limit.H_MIN: parameters["hsv_limit0_H_MIN"],
                                             detect.Limit.H_MAX: parameters["hsv_limit0_H_MAX"],
                                             detect.Limit.S_MIN: parameters["hsv_limit0_S_MIN"],
                                             detect.Limit.S_MAX: parameters["hsv_limit0_S_MAX"],
                                             detect.Limit.V_MIN: parameters["hsv_limit0_V_MIN"],
                                             detect.Limit.V_MAX: parameters["hsv_limit0_V_MAX"]
                                         },
                                         {
                                             detect.Limit.H_MIN: parameters["hsv_limit1_H_MIN"],
                                             detect.Limit.H_MAX: parameters["hsv_limit1_H_MAX"],
                                             detect.Limit.S_MIN: parameters["hsv_limit1_S_MIN"],
                                             detect.Limit.S_MAX: parameters["hsv_limit1_S_MAX"],
                                             detect.Limit.V_MIN: parameters["hsv_limit1_V_MIN"],
                                             detect.Limit.V_MAX: parameters["hsv_limit1_V_MAX"]
                                         }
                                     ])
    else:
        image = detect.rgb_detection(image,
                                     red_weight=parameters["rgb_red_weight"],
                                     rb_offset=parameters["rgb_rb_offset"],
                                     green_weight=parameters["rgb_green_weight"],
                                     sub_weight=parameters["rgb_sub_weight"],
                                     sub_offset=parameters["rgb_sub_offset"])

    image = detect.histogram_adjustment(image,
                                        slope=parameters["hist_slope"],
                                        offset=parameters["hist_offset"])

    image = detect.binarisation(image,
                                threshold=parameters["bin_threshold"])

    image = post.morphologie(image,
                             output,
                             shape_open=parameters["morph_shape_open"],
                             size_open=(parameters["morph_size_open_x"], parameters["morph_size_open_y"]),
                             shape_close=parameters["morph_shape_close"],
                             size_close=(parameters["morph_size_close_x"], parameters["morph_size_close_y"]),
                             open_first=parameters["morph_open_first"])

    return image


def create_process(parameters: dict[str, any])\
        -> Callable[[str | cv2.typing.MatLike, Optional[str]], cv2.typing.MatLike]:
    return lambda image, output = None: __run(image, output, parameters)


def __cross_over_continues(a: any, b: any,
                           lower: any, upper: any,
                           is_int: bool, mutation_chance: float)\
        -> any:
    if np.random.random() <= mutation_chance:
        if is_int:
            return np.random.randint(lower, upper + 1)
        return np.random.random() * (upper - lower) + lower

    mean = (a + b) / 2
    std_div = 2 * (mean - min(a, b))
    c = np.random.normal(mean, std_div)
    if is_int:
        c = int(round(c))
    return max(a, min(b, c))


def __cross_over_categorical(a: any, b: any,
                             values: Sequence[any], mutation_chance: float)\
        -> any:
    if np.random.random() <= mutation_chance:
        return values[np.random.randint(0, len(values))]

    if np.random.random() < 0.5:
        return a
    return b


def __cross_over(parameter: __Parameter, a: any, b: any, mutation_chance: float) -> any:
    if parameter.paramType == _ParameterType.Categorical:
        return __cross_over_categorical(a, b, parameter.values, mutation_chance)
    return __cross_over_continues(a, b, parameter.limits[0], parameter.limits[1], parameter.is_int, mutation_chance)


def cross_over(a: dict[str, any], b: dict[str, any], mutation_chance: float) -> dict[str, any]:
    return {key: __cross_over(__args[key], a[key], b[key], mutation_chance) for key in a}


def __get_random_parameter_value(parameter: __Parameter):
    if parameter.paramType == _ParameterType.Categorical:
        return random.choice(parameter.values)

    lower = parameter.limits[0]
    upper = parameter.limits[1]

    if parameter.is_int:
        return np.random.randint(lower, upper + 1)
    return np.random.random() * (upper - lower) + lower


def get_random_configuration() -> dict[str, any]:
    return {key: __get_random_parameter_value(__args[key]) for key in __args}


def copy_configuration(configuration: dict[str, any]) -> dict[str, any]:
    return {key: configuration[key] for key in configuration}


def to_string(configuration: dict[str, any]):
    return json.dumps(configuration, indent=4)


def load_string(data: str) -> dict[str, any]:
    return json.loads(data)


def get_default_configuration():
    return {key: __args[key].default for key in __args}
