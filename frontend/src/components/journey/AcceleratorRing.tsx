import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export interface AcceleratorRingProps {
  bunchCount: number;
  protonsPerBunch: number;
  journeyState: string;
  onSelectLandmark: (name: string, description: string) => void;
}

export const AcceleratorRing: React.FC<AcceleratorRingProps> = ({
  bunchCount,
  protonsPerBunch,
  journeyState,
  onSelectLandmark,
}) => {
  const pointsRef1 = useRef<THREE.Points>(null!);
  const pointsRef2 = useRef<THREE.Points>(null!);

  const RING_RADIUS = 6.0;
  const particleCount = useMemo(() => {
    // Scale particle count between 200 and 1000 based on bunchCount
    return Math.min(1000, Math.max(200, Math.round((bunchCount / 2808) * 600)));
  }, [bunchCount]);

  // Pre-allocate geometry and particle positions to avoid per-frame allocations
  const { positions1, positions2 } = useMemo(() => {
    const pos1 = new Float32Array(particleCount * 3);
    const pos2 = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const radiusOffset = (Math.random() - 0.5) * 0.1;
      const zOffset = (Math.random() - 0.5) * 0.1;

      // Clockwise beam
      pos1[i * 3] = (RING_RADIUS + radiusOffset) * Math.cos(angle);
      pos1[i * 3 + 1] = zOffset;
      pos1[i * 3 + 2] = (RING_RADIUS + radiusOffset) * Math.sin(angle);

      // Counter-clockwise beam
      pos2[i * 3] = (RING_RADIUS - radiusOffset) * Math.cos(angle);
      pos2[i * 3 + 1] = -zOffset;
      pos2[i * 3 + 2] = (RING_RADIUS - radiusOffset) * Math.sin(angle);
    }
    return { positions1: pos1, positions2: pos2 };
  }, [particleCount]);

  const geo1 = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions1, 3));
    return geo;
  }, [positions1]);

  const geo2 = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions2, 3));
    return geo;
  }, [positions2]);

  // Speed multiplier based on state and beam intensity
  const speedMult = useMemo(() => {
    if (journeyState === 'accelerating' || journeyState === 'colliding') return 4.0;
    if (journeyState === 'injecting') return 2.0;
    return 1.0;
  }, [journeyState]);

  // Frame loop - update point cloud rotation without allocations
  useFrame((_, delta) => {
    if (pointsRef1.current) {
      pointsRef1.current.rotation.y += delta * 0.5 * speedMult;
    }
    if (pointsRef2.current) {
      pointsRef2.current.rotation.y -= delta * 0.5 * speedMult;
    }
  });

  return (
    <group>
      {/* Torus Ring Geometry */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[RING_RADIUS, 0.12, 16, 100]} />
        <meshStandardMaterial
          color="#1E293B"
          metalness={0.8}
          roughness={0.3}
          wireframe={false}
        />
      </mesh>

      {/* Beam Stream 1 (Clockwise - Cyan) */}
      <points ref={pointsRef1} geometry={geo1}>
        <pointsMaterial
          size={0.08}
          color="#06B6D4"
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Beam Stream 2 (Counter-Clockwise - Orange) */}
      <points ref={pointsRef2} geometry={geo2}>
        <pointsMaterial
          size={0.08}
          color="#F97316"
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Landmark 1: Dipole Steering Magnets (-6.0, 0, 0) */}
      <group
        position={[-RING_RADIUS, 0, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectLandmark(
            'Dipole Bending Magnets',
            'Superconducting dipole magnets operating at 8.3 Tesla steer the ultra-relativistic proton bunches along the 27 km circular ring.'
          );
        }}
      >
        <mesh>
          <boxGeometry args={[0.8, 0.8, 1.2]} />
          <meshStandardMaterial color="#3B82F6" metalness={0.6} roughness={0.4} />
        </mesh>
      </group>

      {/* Landmark 2: RF Cavities (0, 0, 6.0) */}
      <group
        position={[0, 0, RING_RADIUS]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectLandmark(
            'Radiofrequency (RF) Cavities',
            'Superconducting RF cavities generate oscillating electromagnetic fields that accelerate proton bunches to 4 TeV beam energy per beam.'
          );
        }}
      >
        <mesh>
          <cylinderGeometry args={[0.5, 0.5, 1.4, 16]} />
          <meshStandardMaterial color="#A855F7" metalness={0.7} roughness={0.3} />
        </mesh>
      </group>

      {/* Landmark 3: ATLAS Detector Shell Interaction Point (6.0, 0, 0) */}
      <group
        position={[RING_RADIUS, 0, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectLandmark(
            'ATLAS Detector Interaction Point',
            'Primary collision point where counter-rotating proton beams collide at 8 TeV center-of-mass energy inside the ATLAS detector.'
          );
        }}
      >
        <mesh>
          <cylinderGeometry args={[0.9, 0.9, 2.0, 24]} />
          <meshStandardMaterial
            color="#06B6D4"
            transparent
            opacity={0.4}
            wireframe
          />
        </mesh>
      </group>
    </group>
  );
};
