import React, { useMemo } from 'react';
import * as THREE from 'three';

interface DetectorSceneProps {
  showAxisHelper?: boolean;
}

/**
 * Renders detector-inspired illustrative geometry, not to scale:
 * - Beam pipe (thin cylinder along z-axis)
 * - Inner tracker (translucent cyan cylinder wireframe)
 * - Outer calorimeter (translucent purple cylinder wireframe)
 * - Optional Cartesian axis helper
 * All geometry elements are educational/illustrative representations.
 */
export const DetectorScene: React.FC<DetectorSceneProps> = ({ showAxisHelper = false }) => {
  // Memoize static Three.js materials to prevent re-allocations on render
  const beamPipeMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#475569', roughness: 0.3, metalness: 0.8 }),
    []
  );

  const trackerMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#06B6D4',
        transparent: true,
        opacity: 0.15,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const calorimeterMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#8B5CF6',
        transparent: true,
        opacity: 0.12,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  return (
    <group>
      {/* Illustrative Beam Pipe along z-axis */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={beamPipeMaterial}>
        <cylinderGeometry args={[0.08, 0.08, 14, 32]} />
      </mesh>

      {/* Illustrative Inner Tracker Shell */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={trackerMaterial}>
        <cylinderGeometry args={[1.2, 1.2, 8, 32, 1, true]} />
      </mesh>

      {/* Illustrative Outer Calorimeter Shell */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={calorimeterMaterial}>
        <cylinderGeometry args={[3.0, 3.0, 10, 32, 1, true]} />
      </mesh>

      {/* Axis Helper if toggled */}
      {showAxisHelper && <axesHelper args={[4]} />}
    </group>
  );
};

export default DetectorScene;
