import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { OrbitControls } from '@react-three/drei';

export type CameraPreset = 'side' | 'transverse' | 'perspective';

export interface CameraControlsProps {
  preset: CameraPreset;
}

export const CameraControls = forwardRef<OrbitControlsImpl, CameraControlsProps>(({ preset }, ref) => {
  const { camera } = useThree();
  const controlsRef = useRef<OrbitControlsImpl>(null);

  useImperativeHandle(ref, () => controlsRef.current!, []);

  useEffect(() => {
    if (!camera || !controlsRef.current) return;

    if (preset === 'side') {
      // Side view looking along y-axis
      camera.position.set(0, 10, 0.01);
    } else if (preset === 'transverse') {
      // Transverse view down z-axis (beam pipe)
      camera.position.set(0, 0.01, 10);
    } else {
      // 3/4 Perspective View (default)
      camera.position.set(7, 6, 7);
    }

    camera.lookAt(0, 0, 0);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  }, [preset, camera]);

  return <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.05} />;
});

CameraControls.displayName = 'CameraControls';
