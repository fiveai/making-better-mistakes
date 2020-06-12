import argparse
import os
import json
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch.optim
import numpy as np

from better_mistakes.data.softmax_cascade import SoftmaxCascade
from better_mistakes.model.init import init_model_on_gpu
from better_mistakes.data.transforms import val_transforms
from better_mistakes.model.run_xent import run
from better_mistakes.model.run_nn import run_nn
from better_mistakes.model.labels import make_all_soft_labels
from better_mistakes.util.label_embeddings import create_embedding_layer
from better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from better_mistakes.util.config import load_config
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes

DATASET_NAMES = ["tiered-imagenet-224", "inaturalist19-224"]


def main(test_opts):
    gpus_per_node = torch.cuda.device_count()

    # Load experiments to run
    with open(os.path.join(test_opts.experiments_path, test_opts.experiments_json)) as fp:
        experiment_to_epochs = json.load(fp)

    for experiment in experiment_to_epochs.keys():
        print("> Results for %s" % experiment)
        epochs_to_run = experiment_to_epochs[experiment]

        expm_json_path = os.path.join(test_opts.experiments_path, experiment, "opts.json")
        assert os.path.isfile(expm_json_path)
        with open(expm_json_path) as fp:
            opts = json.load(fp)
            # convert dictionary to namespace
            opts = argparse.Namespace(**opts)
            opts.out_folder = None
            opts.epochs = 0

        if test_opts.data_path is None:
            opts.data_paths = load_config(test_opts.data_paths_config)
            opts.data_path = opts.data_paths[opts.data]

        # Setup data loaders ------------------------------------------------------------------------------------------
        test_dir = os.path.join(opts.data_path, "test")
        test_dataset = datasets.ImageFolder(test_dir, val_transforms(opts.data, normalize=True))

        # check that classes are loaded in the right order
        def is_sorted(x):
            return x == sorted(x)

        assert is_sorted([d[0] for d in test_dataset.class_to_idx.items()])

        # get data loaders
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers, pin_memory=True, drop_last=True
        )

        # Load hierarchy and classes --------------------------------------------------------------------------------------------------------------------------
        distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
        hierarchy = load_hierarchy(opts.data, opts.data_dir)

        if opts.loss == "yolo-v2":
            classes, _ = get_classes(hierarchy, output_all_nodes=True)
        else:
            classes = test_dataset.classes

        opts.num_classes = len(classes)

        # Model, loss, optimizer ------------------------------------------------------------------------------------------------------------------------------

        # more setup for devise and b+d
        if opts.devise:
            assert not opts.barzdenzler
            embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
            embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
            emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
            assert is_sorted(sorted_keys)

        if opts.barzdenzler:
            assert not opts.devise
            embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
            embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
            emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
            assert is_sorted(sorted_keys)

        # setup loss
        if opts.loss == "cross-entropy":
            loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
        elif opts.loss == "soft-labels":
            loss_function = nn.KLDivLoss().cuda(opts.gpu)
        elif opts.loss == "hierarchical-cross-entropy":
            weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
            loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda(opts.gpu)
        elif opts.loss == "yolo-v2":

            cascade = SoftmaxCascade(hierarchy, classes).cuda(opts.gpu)
            num_leaf_classes = len(hierarchy.treepositions("leaves"))
            weights = get_weighting(hierarchy, "exponential", value=opts.beta)
            loss_function = YOLOLoss(hierarchy, classes, weights).cuda(opts.gpu)

            def yolo2_corrector(output):
                return cascade.final_probabilities(output)[:, :num_leaf_classes]

        elif opts.loss == "cosine-distance":
            loss_function = CosineLoss(emb_layer).cuda(opts.gpu)
        elif opts.loss == "ranking-loss":
            loss_function = RankingLoss(emb_layer, batch_size=opts.batch_size, single_random_negative=opts.devise_single_negative, margin=0.1).cuda(opts.gpu)
        elif opts.loss == "cosine-plus-xent":
            loss_function = CosinePlusXentLoss(emb_layer).cuda(opts.gpu)
        else:
            raise RuntimeError("Unkown loss {}".format(opts.loss))

        # for yolo, we need to decode the output of the classifier as it outputs the conditional probabilities
        corrector = yolo2_corrector if opts.loss == "yolo-v2" else lambda x: x

        # create the solft labels
        soft_labels = make_all_soft_labels(distances, classes, opts.beta)

        # Test ------------------------------------------------------------------------------------------------------------------------------------------------
        summaries, summaries_table = dict(), dict()
        for e in epochs_to_run:

            # setup model
            model = init_model_on_gpu(gpus_per_node, opts)

            checkpoint_id = "checkpoint.epoch%04d.pth.tar" % e
            checkpoint_path = os.path.join(test_opts.experiments_path, experiment, "model_snapshots", checkpoint_id)
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint '{}'".format(checkpoint_path))
            else:
                print("=> no checkpoint found at '{}'".format(checkpoint_path))
                raise RuntimeError

            if opts.devise or opts.barzdenzler:
                summary, _ = run_nn(test_loader, model, loss_function, distances, classes, opts, 0, 0, emb_layer, embeddings_mat, is_inference=True)
            else:
                summary, _ = run(test_loader, model, loss_function, distances, soft_labels, classes, opts, 0, 0, is_inference=True, corrector=corrector)

            for k in summary.keys():
                val = summary[k]
                if "accuracy_top" in k or "ilsvrc_dist_precision" in k or "ilsvrc_dist_mAP" in k:
                    val *= 100
                if "accuracy" in k:
                    k_err = re.sub(r"accuracy", "error", k)
                    val = 100 - val
                    k = k_err
                if k not in summaries:
                    summaries[k] = []
                summaries[k].append(val)

        print("\t\tEpochs: " + ", ".join([str(i) for i in epochs_to_run]))

        for k in summaries.keys():
            avg = np.mean(summaries[k])
            conf95 = 1.96 * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
            summaries_table[k] = (avg, conf95)
            print("\t\t\t\t%20s: %.2f" % (k, summaries_table[k][0]) + " +/- %.4f" % summaries_table[k][1])

        with open(os.path.join(test_opts.experiments_path, experiment, "test_summary.json"), "w") as fp:
            json.dump(summaries, fp, indent=4)
        with open(os.path.join(test_opts.experiments_path, experiment, "test_summary_table.json"), "w") as fp:
            json.dump(summaries_table, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_json", default="experiment_test.json", help="json containing experiments and epochs to run")
    parser.add_argument("--experiments_path", help="Path to experiments logs", default="../experiments")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="../data/", help="Folder containing the supplementary data")
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    test_opts = parser.parse_args()

    main(test_opts)
