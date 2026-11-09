"""
FORK: two particle species on one starting lattice.

With param["baryon_fraction"] = f_b > 0, the particles on the lattice are split:
each 'particle' turns into 2, at the same location: baryons and the second species (the
sterile neutrinos of the v-HDM cosmology). Equal numbers, so with masses relative to the
mean particle mass mpart

    baryon: f_b        other species: (1 - f_b)

which average to exactly 1, so the normalisation of the density (mpart, Om_m) is unchanged.
All particles fall alike: only the density deposit needs these masses (mesh.CIC_weighted).
"""

import numpy as np
import numpy.typing as npt
import pandas as pd


def is_baryon(mass: npt.NDArray[np.float32], param: pd.Series) -> npt.NDArray[np.bool_]:
    """Which particles are baryons, from their relative masses (any particle order)."""
    return np.isclose(mass, float(param["baryon_fraction"]), rtol=1e-5)


def relative_masses(param: pd.Series) -> npt.NDArray[np.float32]:
    """Particle masses relative to the mean, in the lattice order of the initial conditions.

    Parameters
    ----------
    param : pd.Series
        Parameter container. Uses baryon_fraction (absent or 0: one species), npart,
        initial_conditions and output_snapshot_format

    Returns
    -------
    npt.NDArray[np.float32]
        Masses [N_part], or an empty array for one species

    Examples
    --------
    >>> import pandas as pd
    >>> from pysco.species import relative_masses
    >>> param = pd.Series({"baryon_fraction": 0.16, "npart": 64, "initial_conditions": "1LPT", "output_snapshot_format": "HDF5"})
    >>> relative_masses(param).mean()
    """
    f_b = float(param.get("baryon_fraction", 0.0))
    if f_b == 0.0:
        return np.empty(0, dtype=np.float32)
    if not 0.0 < f_b < 1.0:
        raise ValueError(f"{f_b=}, should be in (0, 1)")
    source = param["initial_conditions"]
    if not (isinstance(source, str) and ("lpt" in source.casefold() or source.casefold() == "tophat")):
        raise NotImplementedError(
            f"two species need generated (lattice) initial conditions, not {source=}"
        )
    if param["output_snapshot_format"].casefold() != "hdf5":
        raise NotImplementedError("two species need output_snapshot_format = HDF5")
    n_sites = int(param["npart"])        # lattice sites; each holds one baryon and one neutrino
    return np.concatenate([np.full(n_sites, f_b), np.full(n_sites, 1.0 - f_b)]).astype(np.float32)
