"""Descriptive metrics of actual saved 2D shear-layer fields, not DNS validation."""
from __future__ import annotations
import numpy as np


def shear_metrics(fields, contract, times):
    nx, ny = contract["shape"]
    width, height, _ = contract["geometry_m"]
    dx, dy = width/nx, height/ny
    velocities = np.asarray(fields["U"], float).reshape(-1, ny, nx, 3)
    he = np.asarray(fields["He"], float).reshape(-1, ny, nx)
    if len(velocities) != len(times) or len(he) != len(times):
        raise ValueError("saved shear fields and time axes disagree")
    rows = []
    for t, velocity, composition in zip(times, velocities, he):
        u, v = velocity[..., 0], velocity[..., 1]
        omega = np.gradient(v, dx, axis=1)-np.gradient(u, dy, axis=0)
        stations = []
        for fraction in (.25, .5, .75):
            i = min(nx-1, int(fraction*nx))
            profile = composition[:, i]
            contrast = float(np.ptp(profile))
            z = (profile-profile.min())/contrast if contrast > 1e-12 else np.zeros_like(profile)
            du = np.gradient(u[:, i], dy)
            stations.append(dict(x_m=(i+.5)*dx, He_contrast=contrast,
                composition_width_m=float(np.sum(4*z*(1-z))*dy),
                vorticity_thickness_m=float(np.ptp(u[:, i])/max(np.max(np.abs(du)), 1e-30))))
        rows.append(dict(time=t, stations=stations, He_variance=float(np.var(composition)),
                         integrated_enstrophy=float(.5*np.sum(omega**2)*dx*dy)))
    return dict(schema="shear-layer-observables-v1", samples=rows,
        definition="composition width = integral 4 z(1-z) dy, z uses each station's observed min/max He",
        mesh_time_convergence_verified=False, three_dimensional_turbulence_validated=False)
