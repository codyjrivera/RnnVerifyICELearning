from functools import partial
from itertools import product
from timeit import default_timer as timer

import numpy as np
import pytest
import pickle
import time

from RNN import ICEAlgorithmDriver, Adversarial
from RNN.MultiLayerBase import GurobiMultiLayer
from RNN.ICEMultiLayerBase import ICEMultiLayer

NUM_POINTS=25

points_raw = pickle.load(open("./models/points.pkl", "rb"))
points = points_raw[:NUM_POINTS]

multi_layer_paths = [
                './models/model_20classes_rnn2_fc32_fc32_fc32_fc32_fc32.h5',
                './models/model_20classes_rnn2_rnn2_fc32_fc32_fc32_fc32_fc32.h5',
                './models/model_20classes_rnn4_fc32_fc32_fc32_fc32_fc32.h5',
                './models/model_20classes_rnn4_rnn2_fc32_fc32_fc32_fc32_fc32.h5',
                './models/model_20classes_rnn4_rnn4_fc32_fc32_fc32_fc32_fc32.h5',
                './models/model_20classes_rnn8_fc32_fc32_fc32_fc32_fc32.h5',
              ]

@pytest.mark.timeout(60 * 5)
@pytest.mark.parametrize(['net_path', 'n', 'point_idx'], product(*[multi_layer_paths[1:2], [8], range(len(points))]))
def test_using_multilayer_ice(net_path, n, point_idx):
    print("Evaluating", net_path, "with time =", n, "on input", point_idx)
    point = points[point_idx]
    print(point)
    start = timer()
    method = lambda x: np.argsort(x)[-2]
    gurobi_ptr = partial(ICEMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True, max_steps=4)
    idx_max, other_idx = ICEAlgorithmDriver.get_out_idx(point, n, net_path, method)
    try:
        res, queries_stats, alpha_history = ICEAlgorithmDriver.adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                            gurobi_ptr, n)
        end = timer()
        print("On", net_path, "with time =", n, "on input", point_idx)
        print("The final result was: ", res, "taking", end - start, "s")
    except:
        end = timer()
        print("On", net_path, "with time =", n, "on input", point_idx)
        print("The final result was a timeout or other error, taking", end - start, "s")
    assert True


