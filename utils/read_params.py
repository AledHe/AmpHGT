# read_params.py

# ===============================================================================
# Copyright (C) 2025 by Yongcheng He
# Center for Sustainable Antimicrobials, Department of Pharmacy, College of Veterinary Medicine
# Sichuan Agricultural University, Chengdu

# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# ===============================================================================

import yaml
import os
import re
import ast
from functools import wraps
from datetime import datetime
from collections.abc import MutableMapping

class Cfig(MutableMapping):
    def __init__(self, dictionary, output_dir):
        """
        Initialize the Config object by converting a dictionary into attributes.
        If a value is a dictionary, it recursively converts it into a Config object.
        Additionally, resolve any placeholders in string values.
        """
        self._store = {}
        self.output_dir = output_dir  # Store output_dir for nested Cfig instances
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Cfig(value, output_dir)
            elif isinstance(value, str):
                value = self._resolve_placeholders(value, output_dir)
            self._store[key] = value
            setattr(self, key, value)
    
    def _resolve_placeholders(self, value, output_dir):
        """
        Resolve dynamic placeholders in string values.
        :param value: The string value with potential placeholders.
        :param output_dir: The output directory to use for placeholder substitution.
        :param add more as needed.
        :return: The string with placeholders resolved.
        """
        # Pattern to match ${now:format}
        now_pattern = re.compile(r'\$\{now:(.*?)\}')
        # Replace all occurrences of ${now:...} with the current datetime
        value = now_pattern.sub(lambda m: datetime.now().strftime(m.group(1)), value)
        
        # Replace ${output_dir} with the actual output directory
        value = value.replace("${output_dir}", output_dir)
        
        return value
    
    # MutableMapping abstract methods
    def __getitem__(self, key):
        return self._store[key]
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Cfig(value, self.output_dir)
        elif isinstance(value, str):
            value = self._resolve_placeholders(value, self.output_dir)
        self._store[key] = value
        setattr(self, key, value)
    
    def __delitem__(self, key):
        del self._store[key]
        delattr(self, key)
    
    def __iter__(self):
        return iter(self._store)
    
    def __len__(self):
        return len(self._store)
    
    # Additional helper methods
    def items(self):
        return self._store.items()
    
    def keys(self):
        return self._store.keys()
    
    def values(self):
        return self._store.values()
    
    def to_dict(self):
        result = {}
        for k, v in self._store.items():
            if isinstance(v, Cfig):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

def load_params(file_path, output_dir):
    """
    Load parameters from a YAML file and return a Config object.
    :param file_path: Path to the YAML configuration file.
    :param output_dir: Output directory for resolving placeholders.
    :return: Config object with parameters accessible via attribute notation.
    """
    with open(file_path, 'r') as f:
        print(f"Loading config at {file_path}")
        data = yaml.safe_load(f)
    return Cfig(data, output_dir)

def lcfig(config_path, output_dir):
    """
    Decorator that loads configuration parameters from a YAML file and passes them
    to the decorated function as the first argument.
    :param config_path: Path to the YAML configuration file.
    :param output_dir: Output directory for resolving placeholders.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            dynamic_config = kwargs.pop('config_path', None)
            dynamic_output = kwargs.pop('output_dir', None)
            dynamic_unknown_args = kwargs.pop('unk_args', [])

            final_config_path = dynamic_config or config_path
            final_output_dir = dynamic_output or output_dir
            if not final_config_path:
                raise ValueError("No config_path provided or set in decorator.")
            
            params = load_params(final_config_path, final_output_dir or "logs")
            # print(dynamic_unknown_args)
            overrides = parse_overrides(dynamic_unknown_args)
            if overrides:
                print("[lcfig] Detected overrides:", overrides)
                apply_overrides_to_cfig(params, overrides)

            log_dir = params.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            original_filename = os.path.basename(final_config_path)
            name_part, ext_part = os.path.splitext(original_filename)
            final_config_name = f"{name_part}{ext_part}"  # 如 pretrain.yaml
            final_save_path = os.path.join(log_dir, final_config_name)
            with open(final_save_path, 'w') as f:
                yaml.dump(params.to_dict(), f)
            
            return func(params, *args, **kwargs)
        return wrapper
    return decorator

def parse_value_as_python(value: str):
    """parse str to bool, int, float, list"""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    
def parse_overrides(unknown_args):
    """
    parse unk args to {key: value}
    ["-train*log_interval", "10", "-train*mask_edge", "True"]
    -> {"train.log_interval": 10, "train.mask_edge": True}
    """
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('-'):
            key = arg[1:].replace("*", ".")
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('-'):
                value = parse_value_as_python(unknown_args[i + 1])
                i += 2
            else:
                value = True
                i += 1
            overrides[key] = value
        else:
            i += 1
    return overrides

def apply_overrides_to_cfig(cfg: Cfig, overrides: dict):
    for full_key, value in overrides.items():
        parts = full_key.split('.')
        d = cfg
        for p in parts[:-1]:
            if p not in d:
                raise KeyError(f"Key '{p}' (in '{full_key}') not found in config.")
            d = d[p]
        last = parts[-1]
        if last not in d:
            raise KeyError(f"Key '{full_key}' not found in config.")
        d[last] = value

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, 'output_dir')  # Base output directory

    @lcfig('config.yaml', output_dir)
    def main(cfg):
        # Accessing parameters using attribute notation
        learning_rate = cfg.train.seed
        output_path = cfg.logger.log_dir
        print(f"Learning Rate: {learning_rate}")
        print(f"Output Path: {output_path}")

        # Iterating over training parameters
        print("\nTraining Parameters:")
        for param, value in cfg.train.items():
            print(f"{param}: {value}")

    main()