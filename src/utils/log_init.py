import os
import sys
import datetime

from logging import getLogger, StreamHandler, FileHandler, Formatter
from logging import WARNING, INFO, DEBUG


os.makedirs("results/text_logs/", exist_ok=True)

# Parameters
now = datetime.datetime.now()
save_filename = "results/text_logs/" + now.strftime("%Y%m%d") + ".log"

# default logging format
datefmt = "%Y/%m/%d %H:%M:%S"
default_fmt = Formatter(
    "[%(asctime)s.%(msecs)03d] %(levelname)5s "
    "(%(process)d) %(filename)s: %(message)s",
    datefmt=datefmt
)

# level: CRITICAL > ERROR > WARNING > INFO > DEBUG
logger = getLogger()
logger.setLevel(DEBUG)

try:
    # Rainbow Logging
    from rainbow_logging_handler import RainbowLoggingHandler
    color_msecs = ("green", None, True)
    stream_handler = RainbowLoggingHandler(
        sys.stdout, color_msecs=color_msecs, datefmt=datefmt
    )
    # msecs color
    stream_handler._column_color["."] = color_msecs
    stream_handler._column_color["%(asctime)s"] = color_msecs
    stream_handler._column_color["%(msecs)03d"] = color_msecs
except Exception:
    stream_handler = StreamHandler()

stream_handler.setFormatter(default_fmt)
stream_handler.setLevel(DEBUG)
logger.addHandler(stream_handler)

file_handler = FileHandler(filename=save_filename)
file_handler.setFormatter(default_fmt)
file_handler.setLevel(INFO)
logger.addHandler(file_handler)

# disable pil debug
pil_logger = getLogger("PIL")
pil_logger.setLevel(INFO)

# disable plt debug and info
plt_logger = getLogger("matplotlib")
plt_logger.setLevel(WARNING)
