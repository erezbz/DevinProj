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


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>%(title)s</title>
<style>body{margin:0;background:#0b0e14;color:#cfd6e4;font:13px system-ui,sans-serif}
#bar{position:fixed;left:0;right:0;bottom:0;padding:8px 12px;background:#141925;display:flex;gap:10px;align-items:center}
#s{flex:1}button{background:#2a3350;color:#fff;border:0;padding:5px 12px;border-radius:4px;cursor:pointer}
#leg{position:fixed;top:10px;left:12px}#leg span{display:inline-block;width:10px;height:10px;border-radius:5px;margin:0 4px 0 10px}
canvas{display:block}</style></head><body><canvas id="c"></canvas><div id="leg"></div>
<div id="bar"><button id="p">Pause</button><input id="s" type="range" min="0" value="0"><span id="t"></span></div>
<script>
const D=%(data)s;const C=["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"];
const cv=document.getElementById("c"),g=cv.getContext("2d"),sl=document.getElementById("s"),tl=document.getElementById("t");
sl.max=D.frames.length-1;let f=0,play=true,yaw=0.6,pitch=0.35,drag=null,zoom=1;
document.getElementById("leg").innerHTML=D.title+" "+D.species.map((n,i)=>`<span style="background:${C[i%%10]}"></span>${n}`).join("");
function rs(){cv.width=innerWidth;cv.height=innerHeight-40}addEventListener("resize",rs);rs();
cv.onmousedown=e=>drag=[e.clientX,e.clientY];onmouseup=()=>drag=null;
onmousemove=e=>{if(drag){yaw+=(e.clientX-drag[0])*0.01;pitch+=(e.clientY-drag[1])*0.01;drag=[e.clientX,e.clientY]}};
cv.onwheel=e=>{zoom*=e.deltaY<0?1.1:0.9;e.preventDefault()};
document.getElementById("p").onclick=e=>{play=!play;e.target.textContent=play?"Pause":"Play"};sl.oninput=()=>f=+sl.value;
function draw(){const P=D.frames[f],S=D.sp[f],cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const sc=zoom*Math.min(cv.width,cv.height)/(2.2*D.extent);g.fillStyle="#0b0e14";g.fillRect(0,0,cv.width,cv.height);
const pts=[];for(let i=0;i<S.length;i++){let x=P[3*i]-D.mid[0],y=P[3*i+1]-D.mid[1],z=P[3*i+2]-D.mid[2];
let X=cy*x+sy*z,Z=-sy*x+cy*z,Y=cp*y-sp*Z;Z=sp*y+cp*Z;pts.push([Z,X,Y,S[i]])}pts.sort((a,b)=>a[0]-b[0]);
for(const[Z,X,Y,s]of pts){g.fillStyle=C[s%%10];g.beginPath();g.arc(cv.width/2+X*sc,cv.height/2-Y*sc,D.r,0,7);g.fill()}
tl.textContent="t = "+D.t[f].toPrecision(4)+" s  ("+S.length+" particles)";sl.value=f;
if(play){f=(f+1)%%D.frames.length;yaw+=0.003}requestAnimationFrame(draw)}draw();
</script></body></html>"""


def export_html(trajectory, system, path, max_frames=300, title="accusim", radius=3):
    """Self-contained interactive 3D viewer (drag to rotate, wheel to zoom, timeline slider). No internet needed."""
    import json
    idx = list(range(0, len(trajectory.positions), max(1, len(trajectory.positions) // max_frames)))
    pos = [np.pad(trajectory.positions[i], ((0, 0), (0, 3 - system.dim))) for i in idx]
    allp = np.vstack(pos)
    lo, hi = allp.min(0), allp.max(0)
    if system.box is not None:
        lo, hi = np.zeros(3), np.pad(system.box, (0, 3 - system.dim))
    scale = float(np.max(hi - lo) / 2) or 1.0
    data = {
        "title": title, "species": [sp.name for sp in system.species], "r": radius,
        "t": [float(trajectory.t[i]) for i in idx], "extent": 1.75, "mid": [0, 0, 0],
        "frames": [np.round(((p - (lo + hi) / 2) / scale).ravel(), 4).tolist() for p in pos],
        "sp": [trajectory.species[i].tolist() for i in idx],
    }
    with open(path, "w") as fh:
        fh.write(_HTML % {"title": title, "data": json.dumps(data, separators=(",", ":"))})
