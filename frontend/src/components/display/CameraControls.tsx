import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { OrbitControls } from '@react-three/drei';

export type CameraPreset =
  | 'observatory'
  | 'eventFocus'
  | 'barrelSlice'
  | 'longitudinal'
  | 'inspection'
  | 'side'
  | 'transverse'
  | 'perspective';

export interface CameraControlsProps {
  preset: CameraPreset;
  selectedObjectPosition?: [number, number, number] | null;
  enableIdleDrift?: boolean;
}

export const CameraControls = forwardRef<OrbitControlsImpl, CameraControlsProps>(
  ({ preset, selectedObjectPosition, enableIdleDrift = true }, ref) => {
    const { camera } = useThree();
    const controlsRef = useRef<OrbitControlsImpl>(null);
    const isUserInteractingRef = useRef<boolean>(false);

    useImperativeHandle(ref, () => controlsRef.current!, []);

    // Detect prefers-reduced-motion setting
    const reducedMotionRef = useRef<boolean>(false);
    useEffect(() => {
      if (typeof window !== 'undefined' && window.matchMedia) {
        reducedMotionRef.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      }
    }, []);

    useEffect(() => {
      if (!camera || !controlsRef.current) return;

      if (preset === 'inspection' && selectedObjectPosition) {
        const [x, y, z] = selectedObjectPosition;
        camera.position.set(x + 2.2, y + 1.6, z + 2.2);
        controlsRef.current.target.set(x, y, z);
      } else if (preset === 'eventFocus') {
        camera.position.set(4.5, 3.2, 4.5);
        controlsRef.current.target.set(0, 0, 0);
      } else if (preset === 'barrelSlice' || preset === 'transverse') {
        camera.position.set(0, 0.01, 8.5);
        controlsRef.current.target.set(0, 0, 0);
      } else if (preset === 'longitudinal' || preset === 'side') {
        camera.position.set(0, 8.5, 0.01);
        controlsRef.current.target.set(0, 0, 0);
      } else {
        // 'observatory' or 'perspective' (default 3/4 cutaway view)
        camera.position.set(7.5, 5.5, 7.5);
        controlsRef.current.target.set(0, 0, 0);
      }

      camera.lookAt(controlsRef.current.target);
      controlsRef.current.update();
    }, [preset, selectedObjectPosition, camera]);

    // Perform calm idle camera drift without React state setters or frame allocations
    useFrame((_, delta) => {
      if (
        !enableIdleDrift ||
        reducedMotionRef.current ||
        isUserInteractingRef.current ||
        selectedObjectPosition ||
        !controlsRef.current ||
        !camera
      ) {
        return;
      }
      const angle = delta * 0.025;
      const x = camera.position.x;
      const z = camera.position.z;
      camera.position.x = x * Math.cos(angle) - z * Math.sin(angle);
      camera.position.z = x * Math.sin(angle) + z * Math.cos(angle);
      camera.lookAt(controlsRef.current.target);
      controlsRef.current.update();
    });

    return (
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.05}
        onStart={() => {
          isUserInteractingRef.current = true;
        }}
        onEnd={() => {
          isUserInteractingRef.current = false;
        }}
      />
    );
  }
);

CameraControls.displayName = 'CameraControls';
