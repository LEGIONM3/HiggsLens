import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Line } from '@react-three/drei';
import { RenderableObject } from '../../lib/kinematics';

interface EventObjects3DProps {
  objects: RenderableObject[];
  selectedObjectId: string | null;
  onSelectObject: (obj: RenderableObject) => void;
}

// Pre-allocated static vectors for quaternion calculation to prevent per-render garbage collection
const STATIC_UP = new THREE.Vector3(0, 1, 0);
const STATIC_TARGET = new THREE.Vector3();
const STATIC_QUAT = new THREE.Quaternion();

function getAlignmentEuler(direction: [number, number, number]): THREE.Euler {
  STATIC_TARGET.set(direction[0], direction[1], direction[2]).normalize();
  STATIC_QUAT.setFromUnitVectors(STATIC_UP, STATIC_TARGET);
  return new THREE.Euler().setFromQuaternion(STATIC_QUAT);
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
        const euler = getAlignmentEuler(obj.direction);

        // Mid-point calculations for centering 3D directional vectors
        const endX = obj.direction[0] * obj.length;
        const endY = obj.direction[1] * obj.length;
        const endZ = obj.direction[2] * obj.length;

        const midX = endX / 2;
        const midY = endY / 2;
        const midZ = endZ / 2;

        if (obj.type === 'met') {
          // MET (Missing Transverse Energy): Red dashed vector in transverse xy-plane
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
              {/* Arrow head for MET vector */}
              <mesh position={[endX, endY, endZ]} rotation={euler}>
                <coneGeometry args={[0.15, 0.4, 16]} />
                <meshStandardMaterial color={obj.color} />
              </mesh>
            </group>
          );
        }

        if (obj.type === 'lepton') {
          // Lepton: Cylinder vector aligned with reconstructed eta/phi
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

        // Hadronic Tau or Jets: Cones representing angular shower cone boundaries
        const isTau = obj.type === 'tau';
        const radius = isTau ? 0.15 : 0.45;

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
              side={THREE.DoubleSide}
              emissive={isSelected ? obj.color : '#000000'}
              emissiveIntensity={isSelected ? 0.7 : 0}
            />
          </mesh>
        );
      })}
    </group>
  );
};

export default EventObjects3D;
