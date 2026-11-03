"""The prop silhouette is solid where triangles overlap. Run: python -m tests.test_silhouette"""
import numpy as np
from scipy.spatial.transform import Rotation

from motionbridge.core.fp_points import silhouette

# a camera at the origin looking down -z (the usual Blender camera), 64x64 px
CAM = {'world': np.eye(4).tolist(), 'lens': 50.0, 'sensor': 36.0, 'W': 64, 'H': 64, 'aspect': 1.0}


def cube(size=0.2):
    """A closed cube's 12 triangles about the origin, two per face."""
    s = size / 2
    tris = []
    for k in range(3):
        for side in (-s, s):
            u, v = [i for i in range(3) if i != k]
            quad = []
            for a, b in ((-s, -s), (s, -s), (s, s), (-s, s)):
                p = np.zeros(3)
                p[k], p[u], p[v] = side, a, b
                quad.append(p)
            tris += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    return np.array(tris)


def check():
    tris = cube()
    M = np.eye(4)                           # turned so three faces show and overlap the back ones, 1 m ahead
    M[:3, :3] = Rotation.from_euler('xy', (35, 40), degrees=True).as_matrix()
    M[:3, 3] = (0, 0, -1)
    im = silhouette(tris, M, CAM, (64, 64), scale=1)
    faces = [silhouette(tris[i:i + 2], M, CAM, (64, 64), scale=1) for i in range(0, 12, 2)]
    union = np.logical_or.reduce(faces)
    assert sum(f.any() for f in faces) == 6 and union.sum() > max(f.sum() for f in faces), 'faces should overlap in view'
    assert (im == union).all(), f'{(union & ~im).sum()} px of the faces missing from the silhouette'
    # doubling every triangle (two overlapping copies) changes nothing
    assert (silhouette(np.concatenate([tris, tris]), M, CAM, (64, 64), scale=1) == im).all()
    print(f'ok: cube silhouette solid ({im.sum()} px = the union of its six faces)')


if __name__ == '__main__':
    check()
