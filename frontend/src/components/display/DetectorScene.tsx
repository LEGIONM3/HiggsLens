import React, { useMemo } from 'react';
import * as THREE from 'three';

interface DetectorSceneProps {
  showAxisHelper?: boolean;
}

/**
 * Renders a multi-layered, realistic-looking semi-transparent ATLAS detector cutaway model:
 * - Central Beryllium/Titanium Beam Pipe
 * - Inner Pixel Tracker & SCT Silicon Barrel (Electric Cyan)
 * - Central Solenoid Magnet Shell
 * - Electromagnetic Calorimeter (ECAL) Shell (Teal/Cyan Glass)
 * - Hadronic Calorimeter (HCAL) Shell (Amethyst Glass Wireframe)
 * - Outer Muon Spectrometer Support Rings & Toroid Wheels
 *
 * NOTE: Detector shell and geometry remain illustrative educational visuals (not to scale).
 * Event particle objects are driven by real ATLAS open-data kinematics.
 */
export const DetectorScene: React.FC<DetectorSceneProps> = ({ showAxisHelper = false }) => {
  // Memoize static Three.js materials to ensure 60fps performance without allocations
  const beamPipeMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#64748b',
        roughness: 0.2,
        metalness: 0.9,
      }),
    []
  );

  const pixelTrackerMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#00f0ff',
        transparent: true,
        opacity: 0.2,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const sctTrackerMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#06b6d4',
        transparent: true,
        opacity: 0.14,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const solenoidMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#3b82f6',
        transparent: true,
        opacity: 0.1,
        roughness: 0.1,
        metalness: 0.8,
        side: THREE.DoubleSide,
      }),
    []
  );

  const ecalMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#0ea5e9',
        transparent: true,
        opacity: 0.12,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const hcalMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#8b5cf6',
        transparent: true,
        opacity: 0.1,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const muonRingMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#f59e0b',
        transparent: true,
        opacity: 0.25,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const endcapSupportMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#475569',
        transparent: true,
        opacity: 0.3,
        metalness: 0.7,
        roughness: 0.3,
      }),
    []
  );

  return (
    <group>
      {/* 1. Central Beam Pipe along z-axis */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={beamPipeMaterial}>
        <cylinderGeometry args={[0.08, 0.08, 14, 32]} />
      </mesh>

      {/* 2a. Inner Silicon Pixel Tracker Layer */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={pixelTrackerMaterial}>
        <cylinderGeometry args={[0.6, 0.6, 5, 24, 1, true]} />
      </mesh>

      {/* 2b. Semiconductor Tracker (SCT) Outer Layer */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={sctTrackerMaterial}>
        <cylinderGeometry args={[1.2, 1.2, 7, 32, 1, true]} />
      </mesh>

      {/* 3. Central Solenoid Magnet Shell */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={solenoidMaterial}>
        <cylinderGeometry args={[1.8, 1.8, 7.5, 32, 1, true]} />
      </mesh>

      {/* 4. Electromagnetic Calorimeter (ECAL) Shell */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={ecalMaterial}>
        <cylinderGeometry args={[2.5, 2.5, 9, 36, 2, true]} />
      </mesh>

      {/* 5. Hadronic Tile Calorimeter (HCAL) Outer Shell */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={hcalMaterial}>
        <cylinderGeometry args={[3.6, 3.6, 11, 40, 2, true]} />
      </mesh>

      {/* 6a. Muon Spectrometer Barrel Toroid Ring 1 (z = -4.5) */}
      <mesh position={[0, 0, -4.5]} material={muonRingMaterial}>
        <torusGeometry args={[3.8, 0.08, 16, 48]} />
      </mesh>

      {/* 6b. Muon Spectrometer Barrel Toroid Ring 2 (z = +4.5) */}
      <mesh position={[0, 0, 4.5]} material={muonRingMaterial}>
        <torusGeometry args={[3.8, 0.08, 16, 48]} />
      </mesh>

      {/* 7a. Endcap Support Wheel Left (z = -5.5) */}
      <mesh position={[0, 0, -5.5]} material={endcapSupportMaterial}>
        <ringGeometry args={[0.8, 3.6, 32]} />
      </mesh>

      {/* 7b. Endcap Support Wheel Right (z = +5.5) */}
      <mesh position={[0, 0, 5.5]} material={endcapSupportMaterial}>
        <ringGeometry args={[0.8, 3.6, 32]} />
      </mesh>

      {/* Optional Cartesian Axis Helper */}
      {showAxisHelper && <axesHelper args={[4]} />}
    </group>
  );
};

export default DetectorScene;
