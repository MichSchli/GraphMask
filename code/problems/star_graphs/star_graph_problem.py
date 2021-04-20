import numpy as np

from code.abstract.abstract_test_problem import AbstractTestProblem
from code.utils.evaluation.binary_classification_evaluator import BinaryClassificationEvaluator
import random


class StarGraphProblem(AbstractTestProblem):

    n_types = None
    n_leaves = None

    current_example = 0

    train_set_size = 10000
    dev_set_size = 500
    test_set_size = 500

    def __init__(self, configuration):
        self.n_types = configuration["task"]["n_colours"]
        self.n_leaves = [configuration["task"]["min_leaves"], configuration["task"]["max_leaves"]]

        self.evaluator = BinaryClassificationEvaluator()

        AbstractTestProblem.__init__(self, configuration)

    def count_raw_examples(self, split):
        if split == "train":
            return self.train_set_size
        elif split == "test":
            return self.test_set_size
        else:
            return self.dev_set_size

    def build_example(self, id, split):
        n_leaves = random.randint(*self.n_leaves)
        leaf_srcs = np.arange(1, n_leaves + 1)
        leaf_tgts = np.zeros_like(leaf_srcs)

        edges = np.stack((leaf_srcs, leaf_tgts)).transpose()

        edge_types = np.random.randint(self.n_types, size=len(edges))

        vertex_input = np.ones((edges[-1][0] + 1, 2 * self.n_types), dtype=np.int32) * -1
        xy = np.random.randint(self.n_types, size=2)
        while xy[0] == xy[1]:  # Naively repeat until we have two different numbers
            xy = np.random.randint(self.n_types, size=2)

        for k in range(1, len(vertex_input)):
            vertex_input[k] *= 0
            vertex_input[k][xy[0]] = 1
            vertex_input[k][xy[1] + self.n_types] = 1

        count_x = (edge_types == xy[0]).sum()
        count_y = (edge_types == xy[1]).sum()
        label = count_x > count_y

        attribution_labels = np.zeros_like(leaf_tgts)
        for i in range(len(attribution_labels)):
            if edge_types[i] == xy[0] or edge_types[i] == xy[1]:
                attribution_labels[i] = 1

        return vertex_input, edges, edge_types, label, attribution_labels

    def count_gnn_input_edges(self, batch):
        return sum([len(example[1]) for example in batch])

    def overwrite_labels(self, batch, predictions_to_overwrite_from):
        new_batch = []
        for i in range(len(batch)):
            example = batch[i]
            prediction = predictions_to_overwrite_from[i]
            actual_preds = prediction.get_predictions()[0]

            new_example = [x for x in example]
            new_example[3] = actual_preds

            new_batch.append(new_example)

        return new_batch

    def score_attributions(self, attribution, example):
        tp = 0
        fp = 0
        fn = 0

        attribution_labels = example[4]
        for i in range(len(attribution)):
            if attribution[i] == attribution_labels[i] and attribution_labels[i] == 1:
                tp += 1
            elif attribution[i] != attribution_labels[i] and attribution_labels[i] == 0:
                fp += 1
            elif attribution[i] != attribution_labels[i] and attribution_labels[i] == 1:
                fn += 1

        return tp, fp, fn