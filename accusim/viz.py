"""Matplotlib helpers: animations and energy plots."""
from __future__ import annotations

import numpy as np


def animate(trajectory, system, path, axes=(0, 1), unit=1.0, unit_label="m", fps=20, max_frames=150,
            sizes=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = list(range(0, len(trajectory.positions), max(1, len(trajectory.positions) // max_frames)))
    allpos = np.vstack([trajectory.positions[i] for i in frames]) / unit
    a, b = axes
    pad = 0.05 * (np.ptp(allpos[:, [a, b]]) or 1)
    lo, hi = allpos[:, [a, b]].min(0) - pad, allpos[:, [a, b]].max(0) + pad
    if system.box is not None:
        lo, hi = np.zeros(2), system.box[[a, b]] / unit
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    sc = ax.scatter([], [], s=12)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel(unit_label); ax.set_ylabel(unit_label)
    names = [s.name for s in system.species]
    for i, n in enumerate(names):
        ax.scatter([], [], color=cmap(i % 10), label=n, s=12)
    ax.legend(loc="upper right", fontsize=7)
    text = ax.set_title(title or "")

    def update(k):
        i = frames[k]
        p = trajectory.positions[i] / unit
        sc.set_offsets(p[:, [a, b]])
        sc.set_color([cmap(j % 10) for j in trajectory.species[i]])
        if sizes is not None:
            sc.set_sizes(np.asarray(sizes)[trajectory.species[i]])
        text.set_text(f"{title or ''}  t = {trajectory.t[i]:.3g} s")
        return sc, text

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_energy(trajectory, path, time_unit=1.0, time_label="s"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(trajectory.t) / time_unit
    E = trajectory.total_energy
    fig, ax = plt.subplots(figsize=(6, 3), dpi=90)
    ax.plot(t, (E - E[0]) / (abs(E[0]) or 1), lw=1)
    ax.set_xlabel(time_label); ax.set_ylabel("relative energy error")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def animate3d(trajectory, system, path, unit=1.0, unit_label="m", fps=20, max_frames=120, title=None,
              rotate=True):
    """3D scatter animation with a slowly rotating camera."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if system.dim != 3:
        raise ValueError("animate3d needs a 3D system")
    frames = list(range(0, len(trajectory.positions), max(1, len(trajectory.positions) // max_frames)))
    allpos = np.vstack([trajectory.positions[i] for i in frames]) / unit
    lo, hi = allpos.min(0), allpos.max(0)
    if system.box is not None:
        lo, hi = np.zeros(3), system.box / unit
    half = max(np.max(hi - lo) / 2, 1e-300)
    mid = (lo + hi) / 2
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter([], [], [], s=10)
    for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(mid[k] - half, mid[k] + half)
    ax.set_xlabel(unit_label)
    for i, sp in enumerate(system.species):
        ax.scatter([], [], [], color=cmap(i % 10), label=sp.name, s=10)
    ax.legend(loc="upper right", fontsize=7)

    def update(k):
        i = frames[k]
        p = trajectory.positions[i] / unit
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        sc.set_color([cmap(j % 10) for j in trajectory.species[i]])
        if rotate:
            ax.view_init(elev=25, azim=30 + 120 * k / max(len(frames), 1))
        ax.set_title(f"{title or ''}  t = {trajectory.t[i]:.3g} s")
        return (sc,)

    FuncAnimation(fig, update, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
