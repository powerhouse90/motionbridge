"""Two-view refinement of a GEM capture. GEM reads one view (the front panel) and, being a learned motion model,
pulls rare poses toward ordinary ones: legs flung out in a straddle jump come back bent and short. H3 animates
two panels of the same instant from known cameras (Render Starting Views writes them), so the 2D joints of both
panels pin the limbs down. Per frame, each body bone's rotation (in its parent's frame, so bone lengths stay
exact) is nudged until the joints land on both panels' 2D joints, kept near GEM's pose and near the last frame's
correction; the corrections are then low-passed. The body's place comes from the panels too: GEM's distance from
one camera wobbles (tens of cm a frame), and its path in the room drifts with it, so a dancer in place slides. GEM's
body size is loose from one view (it came out ~10% large on rendered views), so one scale for the take is
fitted first; it only places the joints for the fit (the retarget keeps the character's own proportions).

GEM's in-camera frame is OpenCV (x right, y down, z forward) for the front panel's camera."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from motionbridge.core.smooth import one_euro

BONES = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
         'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftLeg', 'LeftShin', 'LeftFoot', 'RightLeg',
         'RightShin', 'RightFoot')   # bones whose rotation may change
SKIP = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Eye', 'Jaw')  # joints the body fit ignores (fingers, face)
# limbs matched by direction on each panel: a detector's knee or ankle doesn't sit exactly where SOMA's joint does,
# so lengths never agree, but the retarget only uses directions
LIMBS = (('LeftLeg', 'LeftShin'), ('LeftShin', 'LeftFoot'), ('LeftFoot', 'LeftToeBase'), ('RightLeg', 'RightShin'),
         ('RightShin', 'RightFoot'), ('RightFoot', 'RightToeBase'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
         ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'), ('LeftLeg', 'RightLeg'), ('LeftArm', 'RightArm'),
         ('Hips', 'Chest'), ('Chest', 'Neck1'), ('Neck1', 'Head'))
ANGLE = 0.12       # rad: a limb this far off its direction on a panel costs as much as ...
CONF = 0.5         # 2D joints read with less confidence than this are left out
SIGMA = 8.0        # px: (placing the body) a joint this far off its 2D joint
PRIOR = 0.8        # rad: ... a bone turned this far from GEM's pose, or ...
STEP = 0.4         # rad: ... a correction this far from the last frame's
CUTOFF = 6.0       # Hz: the corrections are low-passed (zero phase); fast moves keep their snap


def project(cam, points):
    """Pixels of world points (N, 3) in a starting-view camera (Render Starting Views metadata)."""
    M = np.linalg.inv(np.array(cam['matrix_world']))
    c = points @ M[:3, :3].T + M[:3, 3]
    if cam['projection'] != 'orthographic':
        raise ValueError('two-view refinement expects the orthographic starting views')
    s = max(cam['width'], cam['height']) / cam['ortho_scale']
    return np.c_[cam['width'] / 2 + c[:, 0] * s, cam['height'] / 2 - c[:, 1] * s]


def _locals(world, parents):
    """Local matrices (in the parent's frame) of one frame's world matrices (J, 4, 4); Root is its own."""
    out = world.copy()
    for j in range(1, len(world)):
        out[j] = np.linalg.inv(world[parents[j]]) @ world[j]
    return out


def _fk(local, parents):
    out = local.copy()
    for j in range(1, len(local)):
        out[j] = out[parents[j]] @ local[j]
    return out


def _turned(local, bones, d):
    out = local.copy()
    for k, j in enumerate(bones):
        out[j, :3, :3] = local[j, :3, :3] @ Rotation.from_rotvec(d[3 * k:3 * k + 3]).as_matrix()
    return out


def refine(names, parents, incam, views, front, fps, log=None):
    """incam: (T, J, 4, 4) GEM's in-camera matrices (front panel's camera, OpenCV); views: [(camera, kp2d (T, J-1, 3))]
    with kp2d the panel's 2D joints (pixels, confidence) for joints 1..J-1; front: the front panel's camera.
    Returns (T, len(BONES), 3) rotation corrections (rotation vectors, in each bone's own frame) and a report."""
    names = list(names)
    parents = np.asarray(parents).copy()
    parents[0] = 0
    bones = [names.index(b) for b in BONES]
    fit = [j for j in range(1, len(names)) if not any(s in names[j] for s in SKIP)]
    limbs = np.array([[fit.index(names.index(a)), fit.index(names.index(b))] for a, b in LIMBS])
    Mf = np.array(front['matrix_world'])
    to_world = Mf[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    T = len(incam)

    def placed(P, m):  # in-camera points scaled by m[0] about the camera, then shifted by m[1:] (world)
        return (m[0] * P) @ to_world.T + Mf[:3, 3] + m[1:]

    # one scale for the take (with a shift per sampled frame): GEM's body size against both panels
    sample = list(range(0, T, max(T // 12, 1)))
    base = [(incam[t, fit, :3, 3], [kp[t, np.array(fit) - 1] for _, kp in views]) for t in sample]

    def size_res(x):
        out = []
        for i, (P, kps) in enumerate(base):
            W = placed(P, np.r_[x[0], x[1 + 3 * i:4 + 3 * i]])
            out += [((project(cam, W) - kp[:, :2]) * kp[:, 2:3])[kp[:, 2] >= CONF].ravel() for (cam, _), kp in zip(views, kps)]
        return np.concatenate(out)
    scale = least_squares(size_res, np.r_[1.0, np.zeros(3 * len(sample))], loss='soft_l1', f_scale=20).x[0]
    D = np.zeros((T, len(bones) * 3))
    hips = np.zeros((T, 3))  # where the panels put the hips (in-camera, GEM's units)
    h = names.index('Hips')
    before, after = [], []
    prev = np.zeros(len(bones) * 3)
    for t in range(T):
        local = _locals(incam[t], parents)
        obs = [(cam, kp[t, np.array(fit) - 1]) for cam, kp in views]

        def joints(d):
            return placed(_fk(_turned(local, bones, d[3:]), parents)[fit, :3, 3], np.r_[scale, d[:3]])

        def place_res(d):
            P = joints(d)
            out = [((project(cam, P) - kp[:, :2]) / SIGMA * kp[:, 2:3])[kp[:, 2] >= CONF].ravel() for cam, kp in obs]
            return np.concatenate(out + [d[3:] / PRIOR, (d[3:] - prev) / STEP])

        def res(d):  # each limb's direction on each panel against the read one
            P = joints(d)
            out = []
            for cam, kp in obs:
                u = project(cam, P)
                a, b = limbs[:, 0], limbs[:, 1]
                seen = (kp[a, 2] >= CONF) & (kp[b, 2] >= CONF)
                v, w = u[b] - u[a], kp[b, :2] - kp[a, :2]
                ang = np.arctan2(v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0], (v * w).sum(1))
                out.append((ang / ANGLE * np.minimum(kp[a, 2], kp[b, 2]))[seen])
            return np.concatenate(out + [d[3:] / PRIOR, (d[3:] - prev) / STEP])

        def err(d):  # median limb direction error per panel, degrees
            r = res(d)[:-2 * len(prev)] * ANGLE
            n = [int(((kp[limbs[:, 0], 2] >= CONF) & (kp[limbs[:, 1], 2] >= CONF)).sum()) for _, kp in obs]
            return [np.degrees(np.median(np.abs(r[:n[0]]))), np.degrees(np.median(np.abs(r[n[0]:n[0] + n[1]])))]

        # the whole body placed first (GEM's depth from one view is loose), then the bones
        x0 = np.r_[least_squares(lambda m: place_res(np.r_[m, prev])[:-2 * len(prev)], np.zeros(3), loss='soft_l1').x, prev]
        before.append(err(np.r_[x0[:3], np.zeros(len(prev))]))
        sol = least_squares(lambda d: res(np.r_[x0[:3], d]), x0[3:], loss='soft_l1', f_scale=3.0, max_nfev=600)
        sol.x = np.r_[x0[:3], sol.x]
        D[t] = prev = sol.x[3:]
        after.append(err(sol.x))
        W = placed(incam[t, h:h + 1, :3, 3], np.r_[scale, sol.x[:3]])[0]  # world, then back into GEM's camera
        hips[t] = np.diag([1.0, -1.0, -1.0]) @ Mf[:3, :3].T @ (W - Mf[:3, 3]) / scale
    if T > 3:
        D = one_euro(D, np.zeros(T), fps, CUTOFF, 0)
        hips = one_euro(hips, np.zeros(T), fps, CUTOFF, 0)
    before, after = np.array(before), np.array(after)
    report = dict(before=before, after=after, turn=np.degrees(np.linalg.norm(D.reshape(T, -1, 3), axis=2)), scale=scale, hips=hips)
    if log:
        log(f'two-view refinement: limbs off their direction on the panels (median deg) front {np.median(before[:, 0]):.0f} -> {np.median(after[:, 0]):.0f}, '
            f'second view {np.median(before[:, 1]):.0f} -> {np.median(after[:, 1]):.0f}; bones turned up to {report["turn"].max():.0f} deg')
    return D.reshape(T, len(bones), 3), report


def apply(motion, corrections, incam=None, hips=None):
    """A SomaMotion with each frame's bone corrections (refine()) applied. With GEM's in-camera matrices and the
    panels' hips (refine()'s report['hips']), the hips follow the panels' path, carried into GEM's world by the
    camera-to-world turn of the first frame; else GEM's path is kept."""
    from motionbridge.core.soma import SomaMotion
    parents = motion.parents.copy()
    parents[0] = 0
    bones = [motion.names.index(b) for b in BONES]
    world = motion.world.copy()
    h = motion.names.index('Hips')
    if hips is not None:
        G = motion.world[0, h] @ np.linalg.inv(incam[0, h])  # GEM's camera to its world (still camera)
    for t in range(motion.frames):
        world[t] = _fk(_turned(_locals(motion.world[t], parents), bones, corrections[t].ravel()), parents)
        if hips is not None:
            world[t, :, :3, 3] += G[:3, :3] @ hips[t] + G[:3, 3] - world[t, h, :3, 3]
    return SomaMotion(motion.names, motion.parents, world, motion.rest, motion.fps, motion.source + ' +two-view')
