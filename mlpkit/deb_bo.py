"""Plot bond-order and related quantities from trajectory."""

from ase.io.trajectory import Trajectory


def dbo(traj, i=0, j=1, delta=1, x_distance=True, print_=True, nn=True):
    """Plot bond order between atoms i and j from a trajectory.

    Parameters
    ----------
    traj : str
        Path to the ASE trajectory file.
    i : int
        Index of atom i.
    j : int
        Index of atom j.
    delta : int
        Show the value of Delta.
    x_distance : bool
        Use x-axis as distance (default True).
    print_ : bool
        Print values (default True).
    nn : bool
        Nearest neighbour mode (default True).
    """
    from irff.deb.deb_bde import deb_bo

    images = Trajectory(traj)
    deb_bo(images, i=i, j=j, delta=delta, x_distance=x_distance, print_=print_, nn=nn)