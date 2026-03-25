import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

# Thư mục lưu log
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Đường dẫn file log
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Format log
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _create_handler(log_to_file: bool = True) -> list[logging.Handler]:
    """
    Tạo list handler cho logger (console + file nếu cần).
    """
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handlers.append(console_handler)

    # File handler (xoay vòng, tối đa 5MB, giữ 5 file cũ)
    if log_to_file:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handlers.append(file_handler)

    return handlers


def get_logger(name: Optional[str] = None, log_to_file: bool = True) -> logging.Logger:
    """
    Tạo hoặc lấy logger với cấu hình chuẩn.
    Args:
        name (Optional[str]): Tên logger (thường truyền __name__).
        log_to_file (bool): Có ghi ra file hay không.
    Returns:
        logging.Logger: Logger đã cấu hình.
    """
    logger = logging.getLogger(name if name else "app")
    logger.setLevel(logging.DEBUG)

    # Tránh gắn handler trùng lặp
    if not logger.handlers:
        for h in _create_handler(log_to_file):
            logger.addHandler(h)
        logger.propagate = False

    return logger
