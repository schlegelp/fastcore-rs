import re
from importlib.metadata import version

__version__ = version("navis-fastcore")
__version_vector__ = tuple(
    int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", __version__).groups()
)
