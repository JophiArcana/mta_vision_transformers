import numpy as np
import re
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import torch.nn as nn
import torch.utils.hooks
from torch.utils._pytree import tree_flatten

from infrastructure import utils
from infrastructure.settings import OUTPUT_DEVICE


class Monitor(object):
    def __init__(self, model: nn.Module, output_config: OrderedDict[str, Any], device: str = OUTPUT_DEVICE):
        self.model = model
        self.output_config = output_config
        self.device = device
        
        self.removable_handles: List[torch.utils.hooks.RemovableHandle] = []
        
    def reset(
        self,
        return_mode: Literal["flat", "indices", "array"] = "array",
    ) -> OrderedDict[str, Any]:
        output_dict = OrderedDict()
        self.delete()
        self.add_hooks_to_model(output_dict, return_mode)
        return output_dict
    
    def delete(self) -> None:
        while self.removable_handles:
            self.removable_handles.pop().remove()
    
    @classmethod
    def default_hook_fn(cls, model_: nn.Module, input_: Any, output_: Any) -> Any:
        return tree_flatten(output_)[0][0]
    
    def get_hook_for_output_key(
        self,
        output_key: str,
        output_dict: Dict[str, Any],
        hook_fn: Callable[[nn.Module, Any, Any], Any],
    ) -> Callable[[nn.Module, Any, Any], torch.Tensor]:
        def hook(model_: nn.Module, input_: Any, output_: Any) -> None:
            output_dict.setdefault(output_key, []).append(hook_fn(model_, input_, output_).to(self.device))
        return hook
    
    def get_array_hook_for_output_key(
        self,
        output_key: str,
        output_dict: Dict[str, Any],
        hook_fn: Callable[[nn.Module, Any, Any], Any],
        indices: Tuple[int, ...],
    ) -> Callable[[nn.Module, Any, Any], torch.Tensor]:
        shape = np.array(indices) + 1
        def hook(model_: nn.Module, input_: Any, output_: Any) -> None:
            output_arr: np.ndarray = output_dict.setdefault(output_key, np.empty(shape, dtype=object))
            if np.any(shape > output_arr.shape):
                output_arr = np.pad(output_arr, pad_width=np.stack((
                    np.zeros((len(indices),), dtype=int),
                    np.maximum(shape - output_arr.shape, 0)
                ), axis=1), constant_values=None)
                output_dict[output_key] = output_arr
            
            if output_arr[indices] is None:
                output_arr[indices] = []
            output_arr[indices].append(hook_fn(model_, input_, output_).to(self.device))
        return hook
    
    def add_hooks_to_model(
        self,
        output_dict: OrderedDict[str, Any],
        return_mode: Literal["flat", "indices", "array"],
    ) -> None:
        for attr, metrics in utils.flatten_nested_dict(self.output_config).items():
            if isinstance(metrics, str):
                metrics = (metrics,)
            for metric in metrics:
                if isinstance(metric, str):
                    hook_fn = Monitor.default_hook_fn
                else:
                    metric, hook_fn = metric
                
                regex_components, pre_numeric = [], False
                for subattr in reversed(attr.split(".")):
                    if subattr.isnumeric() or pre_numeric:
                        pre_numeric = not pre_numeric
                        regex_components.append(subattr)
                    else:
                        regex_components.append(f"{subattr}(\\.\\d+|)")
                regex = "^" + "\\.".join(reversed(regex_components)) + "$"
                
                found_modules: Set[str] = set()
                for parameter_name, _ in utils.named_modules(self.model):
                    m = re.match(regex, parameter_name)
                    if m is not None and m.group() not in found_modules:
                        module: nn.Module = utils.rgetattr(self.model, m.group())
                        indices = tuple(int(g.strip(".")) for g in m.groups() if g != "")
                        
                        if return_mode == "flat" or (return_mode == "array" and len(indices) == 0):
                            hook = self.get_hook_for_output_key(metric, output_dict, hook_fn)
                        elif return_mode == "indices":
                            hook = self.get_hook_for_output_key(f"{metric}.{''.join(m.groups())}", output_dict, hook_fn)
                        elif return_mode == "array":
                            hook = self.get_array_hook_for_output_key(metric, output_dict, hook_fn, indices)

                        self.removable_handles.append(
                            module.register_forward_hook(hook)
                        )
                        found_modules.add(f"{metric}:{m.group()}")




