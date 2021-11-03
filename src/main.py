import random

from utils.seed import seed_everything
from utils.gpu import set_visible_gpus, show_gpu_info
from utils.arguments import get_args, load_edict_config
from utils.log_init import logger

from scripts.train_geneva import train_geneva
from scripts.train_propv1_geneva import train_propv1_geneva
from scripts.pretrain_tirg import pretrain_tirg
from scripts.create_embs_from_model import create_embs_from_model
from scripts.train_propv1_scain_geneva import train_propv1_scain_geneva


def main():
    # logger
    logger.info("script start!")

    # arguments --> config
    args = get_args()
    cfg = load_edict_config(args)

    if ("seed" not in cfg) or (cfg.seed is None):
        cfg.seed = random.randint(1000, 2000)

    # seed fix & gpu set
    seed_everything(cfg.seed)
    set_visible_gpus(args.gpu_ids)
    show_gpu_info()

    # --------------------------------------------------
    # scripts
    # --------------------------------------------------

    if args.train_geneva:
        train_geneva(cfg)
    elif args.train_propv1_geneva:
        train_propv1_geneva(cfg)
    elif args.pretrain_tirg:
        pretrain_tirg(cfg)
    elif args.create_embs_from_model:
        create_embs_from_model(cfg)
    elif args.train_propv1_scain_geneva:
        train_propv1_scain_geneva(cfg)


if __name__ == "__main__":
    main()
