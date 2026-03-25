from typing import Any
from xml.etree.ElementTree import Element


def require_attr(node: Element, key: str) -> str:
    """
    Lấy giá trị attribute bắt buộc từ XML element.
    Raise ValueError nếu không có.
    """
    value = node.get(key)
    if value is None:
        raise ValueError(f"Missing required attribute '{key}' in <{node.tag}> element.")
    return value


def require_int_attr(node: Element, key: str) -> int:
    """Lấy attribute kiểu int (bắt buộc)."""
    return int(require_attr(node, key))


def require_float_attr(node: Element, key: str) -> float:
    """Lấy attribute kiểu float (bắt buộc)."""
    return float(require_attr(node, key))
