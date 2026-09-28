#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing_extensions import override

from utils.dataloader import ClassIncrementalLoader, DataPoint, IncrementalLoader


class StrictPerClassIncrementalLoader(IncrementalLoader):
    """
    WRES-safe variant:
    - Prevents X/Y length mismatch when requested per-class samples exceed available samples.
    - Keeps original interface so it can be swapped in without changing downstream code.
    """

    def __init__(
        self,
        data_dir,
        exp_root_dir,
        pretrained_model_name,
        base,
        increment,
        num_labeled,
        num_novel_inc,
        num_known_inc,
        class_order=None,
    ):
        super().__init__(
            data_dir=data_dir,
            exp_root_dir=exp_root_dir,
            pretrained_model_name=pretrained_model_name,
            base=base,
            increment=increment,
        )
        self._increment = increment
        self._base = base
        self.num_labeled = num_labeled
        self.num_novel_inc = num_novel_inc
        self.num_known_inc = num_known_inc
        self._class_order = class_order

        self.cl = ClassIncrementalLoader(
            data_dir=data_dir,
            exp_root_dir=exp_root_dir,
            pretrained_model_name=pretrained_model_name,
            base=base,
            increment=increment,
            class_order=class_order,
        )

    def _resort_data(self, x, y):
        permutation_idx = np.random.permutation(x.shape[0])
        return x[permutation_idx], y[permutation_idx]

    def _sample_known_instance(self, x, y, seen_classes, num_known_inc):
        permutation_idx = np.random.permutation(x.shape[0])
        x = x[permutation_idx]
        y = y[permutation_idx]

        seen_samples_x = []
        seen_samples_y = []
        for c in seen_classes:
            cc_x = x[y == c]
            cc_y = y[y == c]
            take_n = min(num_known_inc, len(cc_x))
            if take_n > 0:
                seen_samples_x.append(cc_x[:take_n])
                seen_samples_y.append(cc_y[:take_n])

        known_x = np.concatenate(seen_samples_x) if seen_samples_x else np.empty((0, x.shape[1]), dtype=x.dtype)
        known_y = np.concatenate(seen_samples_y) if seen_samples_y else np.empty((0,), dtype=y.dtype)
        return x, y, known_x, known_y

    def _train_data_mix(self):
        labeled_per_class = self.num_labeled // self._base
        num_novel_per_stage_per_class = self.num_novel_inc

        unlabeled_x_pool = None
        unlabeled_y_pool = None
        dataset_per_stage = []
        train_loader = self.cl.train_dataloader()
        seen_classes = None

        for stage_i, train_data in enumerate(train_loader):
            if stage_i == 0:
                labeled_x = None
                labeled_y = None

                idx_classes = np.unique(train_data._y)
                seen_classes = idx_classes

                for idx in idx_classes:
                    cls_x = train_data._x[train_data._y == idx]
                    take_n = min(labeled_per_class, len(cls_x))

                    if take_n > 0:
                        labeled_x = self._concatenate_data(labeled_x, cls_x[:take_n])
                        labeled_y = self._concatenate_data(
                            labeled_y,
                            np.full(take_n, idx, dtype=train_data._y.dtype),
                        )

                    unlabeled_x = cls_x[take_n:]
                    unlabeled_y = np.full(len(unlabeled_x), idx, dtype=train_data._y.dtype)
                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                labeled_x, labeled_y = self._resort_data(labeled_x, labeled_y)
                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)
                assert len(labeled_x) == len(labeled_y), "Stage0 X/Y length mismatch"
                dataset_per_stage.append((labeled_x, labeled_y))

            else:
                unlabeled_x_pool, unlabeled_y_pool, known_x, known_y = self._sample_known_instance(
                    unlabeled_x_pool, unlabeled_y_pool, seen_classes, self.num_known_inc
                )

                idx_classes = np.unique(train_data._y)
                unlabeled_novels_x = None
                unlabeled_novels_y = None
                seen_classes = self._concatenate_data(seen_classes, idx_classes)

                for idx in idx_classes:
                    cls_x = train_data._x[train_data._y == idx]
                    take_n = min(num_novel_per_stage_per_class, len(cls_x))

                    if take_n > 0:
                        unlabeled_novels_x = self._concatenate_data(unlabeled_novels_x, cls_x[:take_n])
                        unlabeled_novels_y = self._concatenate_data(
                            unlabeled_novels_y,
                            np.full(take_n, idx, dtype=train_data._y.dtype),
                        )

                    unlabeled_x = cls_x[take_n:]
                    unlabeled_y = np.full(len(unlabeled_x), idx, dtype=train_data._y.dtype)
                    unlabeled_x_pool = self._concatenate_data(unlabeled_x_pool, unlabeled_x)
                    unlabeled_y_pool = self._concatenate_data(unlabeled_y_pool, unlabeled_y)

                stage_x = self._concatenate_data(known_x, unlabeled_novels_x)
                stage_y = self._concatenate_data(known_y, unlabeled_novels_y)
                stage_x, stage_y = self._resort_data(stage_x, stage_y)
                assert len(stage_x) == len(stage_y), f"Stage{stage_i} X/Y length mismatch"
                dataset_per_stage.append((stage_x, stage_y))
                unlabeled_x_pool, unlabeled_y_pool = self._resort_data(unlabeled_x_pool, unlabeled_y_pool)

        return dataset_per_stage

    @override
    def train_dataloader(self):
        train_data_per_stage = self._train_data_mix()
        return map(lambda x: DataPoint(x[0], x[1]), train_data_per_stage)

    @override
    def test_dataloader(self, mode="all"):
        test_loader = self.cl.test_dataloader()

        if mode == "novel":
            return test_loader

        test_data_all_per_stage = []
        test_data_old_per_stage = [(np.array([]), np.array([]))]

        tmp_test_data_x = None
        tmp_test_data_y = None
        for _, test_data in enumerate(test_loader):
            tmp_test_data_x = self._concatenate_data(tmp_test_data_x, test_data._x)
            tmp_test_data_y = self._concatenate_data(tmp_test_data_y, test_data._y)
            test_data_all_per_stage.append((tmp_test_data_x, tmp_test_data_y))
            test_data_old_per_stage.append((tmp_test_data_x, tmp_test_data_y))

        if mode == "all":
            return map(lambda x: DataPoint(x[0], x[1]), test_data_all_per_stage)
        if mode == "old":
            return map(lambda x: DataPoint(x[0], x[1]), test_data_old_per_stage)
        raise ValueError("mode should be 'all' or 'novel'")
