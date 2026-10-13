"""Descriptive metrics of actual saved 2D fields, not DNS/thermal validation."""
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


def crossflow_metrics(fields, contract, times):
    """Resolved planar H2 distribution and vorticity; not a 3D round-jet metric."""
    nx, ny = contract['shape']
    width, height, _ = contract['geometry_m']
    dx, dy = width/nx, height/ny
    h2 = np.asarray(fields['H2'], float).reshape(-1, ny, nx)
    velocity = np.asarray(fields['U'], float).reshape(-1, ny, nx, 3)
    if len(h2) != len(times) or len(velocity) != len(times):
        raise ValueError('saved crossflow fields and time axes disagree')
    y = (np.arange(ny) + .5)*dy
    rows = []
    for t, fraction, u in zip(times, h2, velocity):
        weight = fraction.sum()
        omega = np.gradient(u[..., 1], dx, axis=1)-np.gradient(u[..., 0], dy, axis=0)
        rows.append(dict(time=t, hydrogen_fraction_integral_m2=float(weight*dx*dy),
                         hydrogen_fraction_centroid_y_m=float((fraction*y[:, None]).sum()/weight) if weight > 0 else None,
                         hydrogen_fraction_variance=float(np.var(fraction)),
                         integrated_enstrophy=float(.5*np.sum(omega**2)*dx*dy)))
    return dict(schema='planar-crossflow-observables-v1', samples=rows,
                reacting_flow_validated=False, three_dimensional_jet_validated=False,
                mesh_time_convergence_verified=False)


def sloshing_metrics(fields, contract, times):
    """Volume and centroid retain useful meaning even when an interface folds."""
    nx, ny = contract['shape']
    width, height, depth = contract['geometry_m']
    alpha = np.asarray(fields['alpha.lox'], float).reshape(-1, ny, nx)
    if len(alpha) != len(times):
        raise ValueError('saved sloshing fields and time axes disagree')
    x = (np.arange(nx) + .5)*width/nx
    y = (np.arange(ny) + .5)*height/ny
    rows = []
    for t, fraction in zip(times, alpha):
        weight = fraction.sum()
        rows.append(dict(time=t, liquid_volume_m3=float(weight*width*height*depth/(nx*ny)),
                         liquid_centroid_x_m=float((fraction*x[None, :]).sum()/weight) if weight > 0 else None,
                         liquid_centroid_y_m=float((fraction*y[:, None]).sum()/weight) if weight > 0 else None,
                         alpha_minimum=float(fraction.min()), alpha_maximum=float(fraction.max())))
    return dict(schema='lox-he-mechanical-sloshing-observables-v1', samples=rows,
                gas_phase_approximation='pure_helium', thermal_physics_validated=False,
                phase_change_modeled=False, mesh_time_convergence_verified=False)


def flow_metrics(fields, contract, times):
    from ..h2_fixed_p.data import experiment_kind, CROSSFLOW_KIND, SLOSHING_KIND
    kind = experiment_kind(contract)
    if kind == SLOSHING_KIND:
        return sloshing_metrics(fields, contract, times)
    if kind == CROSSFLOW_KIND:
        return crossflow_metrics(fields, contract, times)
    return shear_metrics(fields, contract, times)
