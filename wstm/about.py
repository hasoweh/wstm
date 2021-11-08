# BEGIN OF LICENSE NOTE
# This file is part of WSTM.
# Copyright (c) 2021, Steve Ahlswede, TU Berlin,
# ahlswede@tu-berlin.de
#
# WSTM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

__version__ = '1.0.0'

__title__ = "TreeSat"
__summary__ = "A Python package for multi-label tree species classification at the image or pixel level."
__uri__ = "https://gitlab.de"

__author__ = "Steve Ahlswede"
__email__ = "ahlswede@tu-berlin.de"

__license__ = "GPLv3+"
__copyright__ = "2021, %s" % __author__


def version():
    """Get the version of WSTM.
    Returns
    -------
    str
        Version specification.
    """
    return __version__