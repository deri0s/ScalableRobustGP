from pathlib import Path
import pandas as pd

file = Path(__file__).resolve()
root = file.parents[0]

def get_nsg_path(file_name: str) -> str:
     return root / 'real_applications/manufacturing/data' / file_name

def get_synthetic_path(file_name: str) -> str:
     return root / 'examples/sine_function' / file_name

def get_motorcycle_path(file_name) -> str:
     return root / 'examples/motorcycle' / file_name

def get_config_path(file_name) -> str:
     return root / 'real_applications/manufacturing/trained' /file_name