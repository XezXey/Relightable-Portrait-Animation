import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

def rotate_sh(cond, src_idx, n_step, axis):
    """
    Rotate spherical harmonics coefficients around a specified axis.
    Args:
        cond (dict): Dictionary containing the spherical harmonics coefficients under the key 'light'.
            - cond['light'] is [1, 27] 
        src_idx (int): Index of the source spherical harmonics coefficients to rotate.
        n_step (int): Number of steps for rotation.
        axis (int): Axis around which to rotate 
            - 0 = top <-> bottom
            - 1 = left <-> right
            - 2 = roundtrip
    """

    import pyshtools as pysh
    def toCoeff(c):
      t = pysh.SHCoeffs.from_zeros(2)
      t.set_coeffs(c[0], 0, 0)
      t.set_coeffs(c[1], 1, 1)
      t.set_coeffs(c[2], 1, -1)
      t.set_coeffs(c[3], 1, 0)
      t.set_coeffs(c[4], 2, -2)
      t.set_coeffs(c[5], 2, 1)
      t.set_coeffs(c[6], 2, -1)
      t.set_coeffs(c[7], 2, 2)
      t.set_coeffs(c[8], 2, 0)
      return t

    def toRGBCoeff(c):
      return [toCoeff(c[::3]), toCoeff(c[1::3]), toCoeff(c[2::3])]

    def toDeca(c):
      a = c.coeffs
      lst = [a[0, 0, 0],
             a[0, 1, 1],
             a[1, 1, 1],
             a[0, 1, 0],
             a[1, 2, 2],
             a[0, 2, 1],
             a[1, 2, 1],
             a[0, 2, 2],
             a[0, 2, 0]]
      return np.array(lst)

    def toRGBDeca(cc):
      return list(itertools.chain(*zip(toDeca(cc[0]), toDeca(cc[1]), toDeca(cc[2]))))

    def axisAngleToEuler(x, y, z, degree):
      xyz = np.array([x, y, z])
      xyz = xyz / np.linalg.norm(xyz)

      rot = R.from_mrp(xyz * np.tan(degree * np.pi / 180 / 4))
      return rot.as_euler('zyz', degrees=True)

    def rotateSH(sh_np, x, y, z, degree):
      cc = toRGBCoeff(sh_np)
      euler = axisAngleToEuler(x, y, z, degree)
      cc[0] = cc[0].rotate(*euler)
      cc[1] = cc[1].rotate(*euler)
      cc[2] = cc[2].rotate(*euler)
      return toRGBDeca(cc)
    
    inp_sh = cond['light'][[src_idx]].flatten()   # [1, 27] -> [27,]
    n = n_step
    out_sh = []
    n = n_step
    for j in np.linspace(0, 360, n):
        moved = rotateSH(inp_sh, axis==0, axis==1, axis==2, j)
        sh_moved = np.array(moved)
        out_sh.append(sh_moved)

    out_sh = np.stack(out_sh, 0)    # [n_step, 27]
    return {'light':out_sh}
  
def interp_sh(cond, n_step):
  # Lerp interpolation between 2 light
  source_light = cond['source_light'][0]
  target_light = cond['target_light'][0]
  lerping = np.linspace(0, 1, n_step)
  out_sh = []
  for l in lerping:
    out_sh.append((1 - l) * source_light + l * target_light)
  return {'light': np.array(out_sh)}