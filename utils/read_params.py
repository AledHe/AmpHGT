# read_params.py

import yaml
import os
import re
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
            # 从 kwargs 提取覆盖参数
            dynamic_config = kwargs.pop('config_path', None)
            dynamic_output = kwargs.pop('output_dir', None)
            
            # 确定最终使用的路径
            final_config_path = dynamic_config or config_path
            final_output_dir = dynamic_output or output_dir
            
            # 加载配置
            params = load_params(final_config_path, final_output_dir)
            
            # 构建保存路径
            log_dir = params.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成带 _final 后缀的 config 文件名
            original_filename = os.path.basename(final_config_path)  # 如 pretrain.yaml
            name_part, ext_part = os.path.splitext(original_filename)  # (pretrain, .yaml)
            final_config_name = f"{name_part}{ext_part}"  # pretrain.yaml
            final_save_path = os.path.join(log_dir, final_config_name)
            
            # 保存（确保不覆盖原文件）
            with open(final_save_path, 'w') as f:
                yaml.dump(params.to_dict(), f)
            
            return func(params, *args, **kwargs)
        return wrapper
    return decorator

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