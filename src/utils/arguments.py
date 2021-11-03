import argparse
import yaml

from easydict import EasyDict as edict


def get_args():
    """Get args which has path to yaml.

    Returns
    -------
    args : NamedTuple
        args from argparse to load yaml file.
    """
    parser = argparse.ArgumentParser()

    # --------------------------------------------------
    # configurations
    # --------------------------------------------------

    parser.add_argument(
        "--yaml_path", type=str, default=None, required=True,
        help="yaml file path",
    )

    parser.add_argument(
        "--gpu_ids", type=str, default="0",
        help="comma separated gpu ids",
    )

    # --------------------------------------------------
    # scripts
    # --------------------------------------------------

    parser.add_argument(
        "--train_geneva", action="store_true",
        help="train geneva",
    )
    parser.add_argument(
        "--train_propv1_geneva", action="store_true",
        help="train proposal version.1 geneva",
    )
    parser.add_argument(
        "--pretrain_tirg", action="store_true",
        help="pretrain tirg network by pairwise ranking loss",
    )
    parser.add_argument(
        "--create_embs_from_model", action="store_true",
        help="create embedding as additional h5 file using pretrained sentence encoder",
    )
    parser.add_argument(
        "--train_propv1_scain_geneva", action="store_true",
        help="train proposal version.1 with scain geneva",
    )

    args = parser.parse_args()

    return args


def load_edict_config(args):
    """Load config in yaml specified in args.yaml_path.

    Parameters
    ----------
    args : NamedTuple
        MUST have args.yaml_path.

    Returns
    -------
    config : EasyDict
        EasyDict which has all parameters written in yaml file.
    """
    with open(args.yaml_path, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    config = edict(config)

    return config
