import sys

from .__version__ import __version__, __version_vector__

from navis_fastcore import _fastcore
from .caps import *
from .cmtk import *
from .dag import *
from .downsample import *
from .elastix import *
from .linkage import *
from .matches import *
from .mesh import *
from .nblast import *
from .threads import *
from .topo import *
from .warp import *

# The interface is one flat namespace — `fastcore.geodesic_matrix`, not
# `fastcore.mesh.geodesic_matrix` — composed from the submodules' own `__all__`
# rather than written out again, so a new function is exported by listing it in
# one place.
#
# Via `sys.modules` rather than `from . import mesh`, because the star-imports
# above bind the *functions* `linkage` and `nblast` over the submodules of the
# same name — so an attribute lookup would hand back the function.
#
# Without any of this, `from navis_fastcore import *` would take the default:
# every non-underscore name bound here, which drags in the submodule objects and
# drops `__version__` for starting with an underscore.
__all__ = ["__version__", "__version_vector__"]
for _sub in (
    "caps",
    "cmtk",
    "dag",
    "downsample",
    "elastix",
    "linkage",
    "matches",
    "mesh",
    "nblast",
    "threads",
    "topo",
    "warp",
):
    __all__ += sys.modules[f"{__name__}.{_sub}"].__all__
del _sub
