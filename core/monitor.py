import numpy as np
import re
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch.nn as nn
import torch.utils.hooks
from torch.utils._pytree import tree_flatten

from infrastructure import utils


class Monitor(object):
    def __init__(self, model: nn.Module, output_config: OrderedDict[str, Any]):
        self.model = model
        self.output_config = output_config
        
        self.removable_handles: List[torch.utils.hooks.RemovableHandle] = []
        
    def reset(
        self,
        return_mode: Literal["flat", "indices", "array"] = "array",
    ) -> OrderedDict[str, Any]:
        output_dict = OrderedDict()
        for handle in self.removable_handles:
            handle.remove()
        self.add_hooks_to_vision_model(output_dict, return_mode)
        return output_dict
    
    @classmethod
    def default_hook_fn(cls, model_: nn.Module, input_: Any, output_: Any) -> Any:
        return tree_flatten(output_)[0][0]
    
    @classmethod
    def get_hook_for_output_key(
        cls,
        output_key: str,
        output_dict: Dict[str, Any],
        hook_fn: Callable[[nn.Module, Any, Any], Any],
    ) -> Callable[[nn.Module, Any, Any], None]:
        def hook(model_: nn.Module, input_: Any, output_: Any) -> None:
            output_dict.setdefault(output_key, []).append(hook_fn(model_, input_, output_))
        return hook
    
    @classmethod
    def get_array_hook_for_output_key(
        cls,
        output_key: str,
        output_dict: Dict[str, Any],
        hook_fn: Callable[[nn.Module, Any, Any], Any],
        indices: Tuple[int, ...],
    ):
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
            output_arr[indices].append(hook_fn(model_, input_, output_))
        return hook
    
    def add_hooks_to_vision_model(
        self,
        output_dict: OrderedDict[str, Any],
        return_mode: Literal["flat", "indices", "array"],
    ) -> List[torch.utils.hooks.RemovableHandle]:
        
        removable_handles: List[torch.utils.hooks.RemovableHandle] = []
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
                regex = "\\.".join(reversed(regex_components))
                
                found_modules = set()
                for parameter_name, _ in utils.named_modules(self.model):
                    m = re.match(regex, parameter_name)
                    if m is not None and m.group() not in found_modules:
                        module = utils.rgetattr(self.model, m.group())
                        indices = tuple(int(g.strip(".")) for g in m.groups() if g != "")
                        
                        if return_mode == "flat" or (return_mode == "array" and len(indices) == 0):
                            hook = Monitor.get_hook_for_output_key(metric, output_dict, hook_fn)
                        elif return_mode == "indices":
                            hook = Monitor.get_hook_for_output_key(f"{metric}.{''.join(m.groups())}", output_dict, hook_fn)
                        elif return_mode == "array":
                            hook = Monitor.get_array_hook_for_output_key(metric, output_dict, hook_fn, indices)

                        removable_handles.append(
                            module.register_forward_hook(hook)
                        )
                        found_modules.add(f"{metric}:{m.group()}")
                    
        return removable_handles
    
    





