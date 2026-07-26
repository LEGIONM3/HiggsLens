import React from 'react';
import * as THREE from 'three';
import { Line } from '@react-three/drei';
import { RenderableObject } from '../../lib/kinematics';

interface EventObjects3DProps {
  objects: RenderableObject[];
  selectedObjectId: string | null;
  onSelectObject: (obj: RenderableObject) => void;
}

/**
 * Calculates a quaternion orientation that aligns a default upward geometry [0, 1, 0]
 * with a target 3D direction vector [dx, dy, dz].
 */
function getAlignmentQuaternion(direction: [number, number, number]): THREE.Quaternion {
  const target = new THREE.Vector3(direction[0], direction[1], direction[2]).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(up, target);
  return q;
}

export const EventObjects3D: React.FC<EventObjects3DProps> = ({
  objects,
  selectedObjectId,
  onSelectObject,
}) => {
  return (
    <group>
      {objects.map((obj) => {
        const isSelected = selectedObjectId === obj.id;
        const q = getAlignmentQuaternion(obj.direction);
        const euler = new THREE.Euler().setFromQuaternion(q);

        // Calculate mid-point position along vector for cone/cylinder centering
        const endX = obj.direction[0] * obj.length;
        const endY = obj.direction[1] * obj.length;
        const endZ = obj.direction[2] * obj.length;

        const midX = endX / 2;
        const midY = endY / 2;
        const midZ = endZ / 2;

        if (obj.type === 'met') {
          // MET: Red dashed line in transverse plane using Drei Line + arrow head
          return (
            <group key={obj.id} onClick={(e) => { e.stopPropagation(); onSelectObject(obj); }}>
              <Line
                points={[[0, 0, 0], [endX, endY, endZ]]}
                color={obj.color}
                dashed
                dashSize={0.3}
                gapSize={0.15}
                lineWidth={3}
              />
              {/* Arrow tip for MET */}
              <mesh position={[endX, endY, endZ]} rotation={euler}>
                <coneGeometry args={[0.15, 0.4, 16]} />
                <meshStandardMaterial color={obj.color} />
              </mesh>
            </group>
          );
        }

        if (obj.type === 'lepton') {
          // Lepton: Cyan thin cylinder track
          return (
            <mesh
              key={obj.id}
              position={[midX, midY, midZ]}
              rotation={euler}
              onClick={(e) => { e.stopPropagation(); onSelectObject(obj); }}
            >
              <cylinderGeometry args={[0.04, 0.04, obj.length, 16]} />
              <meshStandardMaterial
                color={obj.color}
                emissive={isSelected ? obj.color : '#000000'}
                emissiveIntensity={isSelected ? 0.6 : 0}
              />
            </mesh>
          );
        }

        // Hadronic Tau or Jets: Cones
        const isTau = obj.type === 'tau';
        const radius = isTau ? 0.15 : 0.45; // Narrow cone for tau, wider for jets

        return (
          <mesh
            key={obj.id}
            position={[midX, midY, midZ]}
            rotation={euler}
            onClick={(e) => { e.stopPropagation(); onSelectObject(obj); }}
          >
            <coneGeometry args={[radius, obj.length, 32, 1, true]} />
            <meshStandardMaterial
              color={obj.color}
              transparent
              opacity={0.85}
              side={2}
              emissive={isSelected ? obj.color : '#000000'}
              emissiveIntensity={isSelected ? 0.7 : 0}
            />
          </mesh>
        );
      })}
    </group>
  );
};
