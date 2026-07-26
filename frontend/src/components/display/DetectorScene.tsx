import React from 'react';

interface DetectorSceneProps {
  showAxisHelper?: boolean;
}

/**
 * Renders stylized illustrative detector geometry:
 * - Beam pipe (thin cylinder along z-axis)
 * - Inner tracker (translucent cyan cylinder)
 * - Outer calorimeter (translucent purple cylinder)
 * - Optional Cartesian axis helper
 */
export const DetectorScene: React.FC<DetectorSceneProps> = ({ showAxisHelper = false }) => {
  return (
    <group>
      {/* Beam Pipe along z-axis */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.08, 0.08, 14, 32]} />
        <meshStandardMaterial color="#475569" roughness={0.3} metalness={0.8} />
      </mesh>

      {/* Inner Tracker Cylinder */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.2, 1.2, 8, 32, 1, true]} />
        <meshStandardMaterial
          color="#06B6D4"
          transparent
          opacity={0.15}
          wireframe
          side={2} // DoubleSide
        />
      </mesh>

      {/* Outer Calorimeter Cylinder */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[3.0, 3.0, 10, 32, 1, true]} />
        <meshStandardMaterial
          color="#8B5CF6"
          transparent
          opacity={0.12}
          wireframe
          side={2} // DoubleSide
        />
      </mesh>

      {/* Axis Helper if toggled */}
      {showAxisHelper && <axesHelper args={[4]} />}
    </group>
  );
};
