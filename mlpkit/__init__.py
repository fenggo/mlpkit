"""MLPKit — An ensemble of toolkit for machine learning potential."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "pred":
        from mlpkit.core import pred as _pred
        return _pred
    if name == "calc":
        from mlpkit.core import calc as _calc
        return _calc
    if name == "traj":
        from mlpkit.core import traj as _traj
        return _traj
    if name == "zmat":
        from mlpkit.core import zmat as _zmat
        return _zmat
    if name == "fdf":
        from mlpkit.core import fdf as _fdf
        return _fdf
    if name == "sample":
        from mlpkit.core import sample as _sample
        return _sample
    if name == "info":
        from mlpkit.core import info as _info
        return _info
    if name == "fingerprint":
        from mlpkit.core import fingerprint as _fingerprint
        return _fingerprint
    if name == "lib":
        from mlpkit.core import lib as _lib
        return _lib
    if name == "ffield":
        from mlpkit.core import ffield as _ffield
        return _ffield
    if name == "molinfo":
        from mlpkit.core import molinfo as _molinfo
        return _molinfo
    if name == "md2pdf":
        from mlpkit.md2pdf import md2pdf as _md2pdf
        return _md2pdf
    raise AttributeError(f"module 'mlpkit' has no attribute {name!r}")
