#!/usr/bin/env python3
"""
Lượng tử hóa INT8 cho TinyML Edge Engine ESP32-S3.
Wrapper gọi export_multisensor_int8 từ quantize_multisensor_int8.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mlops.scripts.quantize_multisensor_int8 import export_multisensor_int8

if __name__ == "__main__":
    export_multisensor_int8()
