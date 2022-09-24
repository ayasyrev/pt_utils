import importlib
from importlib.util import find_spec
from typing import Any, Callable, Dict


lib_to_package = {
    "PIL": "pillow",
    "accimage": "accimage",
    "jpeg4py": "jpeg4py",

}

img_libs_supported = list(lib_to_package)

img_libs_available = [
    lib_name for lib_name in lib_to_package if find_spec(lib_name) is not None
]


def get_loader(lib_name: str = "pil", module_name: str = "pt_utils.data") -> Any:
    """Load img loader for lib."""
    module = importlib.import_module(f"{module_name}.img_loader_{lib_name.lower()}")
    return getattr(module, f"loader_{lib_name}", None)


loaders: Dict[str, Callable[[str], Any]] = {}  # no comprehension in favor python 3.7 - no walrus operation
for lib_name in img_libs_available:
    loader = get_loader(lib_name.lower())
    if loader is not None:
        loaders[lib_name] = loader
