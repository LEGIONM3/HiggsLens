import React, { useMemo } from 'react';
import * as THREE from 'three';

interface DetectorSceneProps {
  showAxisHelper?: boolean;
}

/**
 * Release B: Orbital Scientific Observatory Detector Scene
 * Renders a multi-layered ATLAS detector cutaway visualization with:
 * - 80-90 degree structural cutaway wedge exposing internal tracker & beam pipe
 * - Deep volumetric atmospheric fog and faint procedural star-dust field
 * - Transverse radial measurement grid floor and beam-axis scale markers
 * - Interaction Point (IP) origin indicator
 *
 * NOTE: Detector shell, grid, fog, and scale markers are illustrative visual design elements.
 * Event particle objects remain 100% data-driven from real ATLAS open-data recorded kinematics.
 */
export const DetectorScene: React.FC<DetectorSceneProps> = ({ showAxisHelper = false }) => {
  // 1. Cutaway wedge parameters (leaves an ~80 degree open window facing 3/4 view)
  const cutawayStart = Math.PI * 0.2;
  const cutawayLength = Math.PI * 1.55;

  // 2. Memoize Three.js materials to prevent re-allocations during render frames
  const beamPipeMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#334155',
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
        opacity: 0.25,
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
        color: '#1e3a8a',
        transparent: true,
        opacity: 0.12,
        roughness: 0.1,
        metalness: 0.8,
        side: THREE.DoubleSide,
      }),
    []
  );

  const ecalMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#0284c7',
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
        color: '#7c3aed',
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
        color: '#d97706',
        transparent: true,
        opacity: 0.3,
        wireframe: true,
        side: THREE.DoubleSide,
      }),
    []
  );

  const gridLineMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: '#1e293b',
        transparent: true,
        opacity: 0.5,
      }),
    []
  );

  const ipMarkerMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#00f0ff',
        transparent: true,
        opacity: 0.6,
        wireframe: true,
      }),
    []
  );

  // 3. Generate static star-dust particles in space [-30..30]
  const stardustPositions = useMemo(() => {
    const count = 250;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 60;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 60;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 60;
    }
    return pos;
  }, []);

  return (
    <group>
      {/* 1. Spatial Observatory Deep Atmospheric Fog & Background */}
      <color attach="background" args={['#070b12']} />
      <fog attach="fog" args={['#070b12', 14, 40]} />

      {/* 2. Procedural Star-Dust Particle Field */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[stardustPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial size={0.06} color="#38bdf8" transparent opacity={0.22} sizeAttenuation />
      </points>

      {/* 3. Central Beryllium Beam Pipe along z-axis */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={beamPipeMaterial}>
        <cylinderGeometry args={[0.08, 0.08, 14, 32]} />
      </mesh>

      {/* 4a. Inner Silicon Pixel Tracker Layer with Cutaway */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={pixelTrackerMaterial}>
        <cylinderGeometry args={[0.6, 0.6, 5, 24, 1, true, cutawayStart, cutawayLength]} />
      </mesh>

      {/* 4b. Semiconductor Tracker (SCT) Outer Layer with Cutaway */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={sctTrackerMaterial}>
        <cylinderGeometry args={[1.2, 1.2, 7, 32, 1, true, cutawayStart, cutawayLength]} />
      </mesh>

      {/* 5. Central Solenoid Magnet Shell with Cutaway */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={solenoidMaterial}>
        <cylinderGeometry args={[1.8, 1.8, 7.5, 32, 1, true, cutawayStart, cutawayLength]} />
      </mesh>

      {/* 6. Electromagnetic Calorimeter (ECAL) Shell with Cutaway */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={ecalMaterial}>
        <cylinderGeometry args={[2.5, 2.5, 9, 36, 2, true, cutawayStart, cutawayLength]} />
      </mesh>

      {/* 7. Hadronic Tile Calorimeter (HCAL) Shell with Cutaway */}
      <mesh rotation={[Math.PI / 2, 0, 0]} material={hcalMaterial}>
        <cylinderGeometry args={[3.6, 3.6, 11, 40, 2, true, cutawayStart, cutawayLength]} />
      </mesh>

      {/* 8a. Outer Muon Toroid Ring Left (z = -4.5) */}
      <mesh position={[0, 0, -4.5]} material={muonRingMaterial}>
        <torusGeometry args={[3.8, 0.08, 16, 48, cutawayLength]} />
      </mesh>

      {/* 8b. Outer Muon Toroid Ring Right (z = +4.5) */}
      <mesh position={[0, 0, 4.5]} material={muonRingMaterial}>
        <torusGeometry args={[3.8, 0.08, 16, 48, cutawayLength]} />
      </mesh>

      {/* 9. Interaction Point (IP) Origin Marker */}
      <mesh position={[0, 0, 0]} material={ipMarkerMaterial}>
        <sphereGeometry args={[0.12, 16, 16]} />
      </mesh>

      {/* 10. Transverse Measurement Grid Floor (y = -3.8) */}
      <group position={[0, -3.8, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        {[1.5, 3.0, 4.5, 6.0].map((radius) => (
          <mesh key={radius} material={gridLineMaterial}>
            <ringGeometry args={[radius - 0.02, radius + 0.02, 64]} />
          </mesh>
        ))}
      </group>

      {/* Optional Cartesian Axis Helper */}
      {showAxisHelper && <axesHelper args={[4]} />}
    </group>
  );
};

export default DetectorScene;
