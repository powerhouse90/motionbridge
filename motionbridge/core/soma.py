"""Canonical motion: the SOMA 77-joint skeleton (plus a static root) in Y-up meters.

Every motion source writes a SomaMotion and every later stage reads one. Only
world-space joint matrices are stored; local rotations are derived when needed.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = 'motionbridge.soma.v1'


@dataclass(frozen=True)
class SomaMotion:
    names: tuple            # 78 joint names; index 0 is the static Root
    parents: np.ndarray     # (78,) int, -1 for Root
    world: np.ndarray       # (T, 78, 4, 4) joint-to-world matrices, Y-up, meters
    rest: np.ndarray        # (78, 4, 4) rest pose, same convention
    fps: float
    source: str = ''

    def __post_init__(self):
        n = len(self.names)
        if self.parents.shape != (n,) or self.rest.shape != (n, 4, 4):
            raise ValueError('names, parents and rest disagree on joint count')
        if self.world.ndim != 4 or self.world.shape[1:] != (n, 4, 4):
            raise ValueError(f'world must be (T, {n}, 4, 4), got {self.world.shape}')
        if self.fps <= 0:
            raise ValueError('fps must be positive')

    @property
    def frames(self):
        return self.world.shape[0]

    @property
    def positions(self):
        return self.world[:, :, :3, 3]

    def index(self, name):
        return self.names.index(name)

    def save(self, path):
        np.savez_compressed(path, format=FORMAT, names=np.array(self.names), parents=self.parents,
                            world=self.world.astype(np.float32), rest=self.rest.astype(np.float32),
                            fps=self.fps, source=self.source)


def load(path):
    """Read a file written by SomaMotion.save."""
    d = np.load(path)
    if str(d['format']) != FORMAT:
        raise ValueError(f'{path} is not a {FORMAT} file')
    return SomaMotion(tuple(d['names'].tolist()), d['parents'].astype(int), d['world'].astype(np.float64),
                      d['rest'].astype(np.float64), float(d['fps']), str(d['source']))


def from_gem_export(path, fps, space='global'):
    """Read GEM's motion_export.npz. GEM does not record frame rate, so pass it."""
    d = np.load(path)
    world = d[space + '_transforms'].astype(np.float64)
    joints = d[space + '_joints']
    # GEM's joint array omits the static Root; check the matrices agree with it.
    error = np.abs(world[:, 1:, :3, 3] - joints).max()
    if error > 1e-4:
        raise ValueError(f'{path}: transforms and joints disagree by {error:.6f} m')
    parents = d['parents'].astype(int).copy()
    parents[0] = -1
    return SomaMotion(tuple(d['names'].tolist()), parents, world, d['reference_transforms'][0].astype(np.float64),
                      float(fps), f'gem:{Path(path).as_posix()}')
