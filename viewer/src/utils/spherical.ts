/**
 * Spherical → Cartesian conversion for Three.js (Y-up).
 *
 * AFE stores arrows as (θ azimuth, φ elevation, r magnitude) in the
 * standard math/data-science convention where Z is up.
 *
 * Math convention (Z-up):
 *   x = r * cos(φ) * cos(θ)
 *   y = r * cos(φ) * sin(θ)
 *   z = r * sin(φ)
 *
 * Three.js convention (Y-up) — swap Y↔Z:
 *   x = r * cos(φ) * cos(θ)       (same)
 *   y = r * sin(φ)                 (elevation → Y up)
 *   z = r * cos(φ) * sin(θ)       (horizontal → Z forward)
 */

import * as THREE from 'three';

const _UP = new THREE.Vector3(0, 1, 0);
const _tmpDir = new THREE.Vector3();

/**
 * Convert AFE spherical (θ, φ, r) to Three.js Cartesian (Y-up).
 * Returns [dx, dy, dz] direction vector scaled by magnitude r.
 */
export function sphericalToThreeJS(
  theta: number,
  phi: number,
  r: number,
): [number, number, number] {
  const cosPhi = Math.cos(phi);
  return [
    r * cosPhi * Math.cos(theta), // X
    r * Math.sin(phi),            // Y (up in Three.js)
    r * cosPhi * Math.sin(theta), // Z (forward in Three.js)
  ];
}

/**
 * Compute a quaternion that rotates the default cone direction (+Y)
 * to the given arrow direction vector.
 *
 * Handles the anti-parallel edge case (direction ≈ -Y) by rotating
 * π around the X axis.
 */
export function directionQuaternion(
  dx: number,
  dy: number,
  dz: number,
  out?: THREE.Quaternion,
): THREE.Quaternion {
  const q = out ?? new THREE.Quaternion();
  _tmpDir.set(dx, dy, dz);
  const len = _tmpDir.length();
  if (len < 1e-8) {
    // Zero-length arrow — identity rotation
    return q.identity();
  }
  _tmpDir.divideScalar(len);

  // Check for anti-parallel case (dot ≈ -1)
  const dot = _UP.dot(_tmpDir);
  if (dot < -0.9999) {
    // Rotate π around X axis
    q.set(1, 0, 0, 0); // (x, y, z, w) = 180° around X
    return q;
  }

  return q.setFromUnitVectors(_UP, _tmpDir);
}
