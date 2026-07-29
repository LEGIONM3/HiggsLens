import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { Play, Pause, RotateCcw, Camera, ArrowRight, Activity, Gauge } from 'lucide-react';
import { createDeterministicPRNG, DEFAULT_BEAM_CONFIG } from '../../../lib/journeyState';
import { NonWebGLPolarView } from '../fallback/NonWebGLPolarView';

type CameraPreset = 'tube' | 'observatory' | 'collision' | 'longitudinal' | 'free';

interface BeamPacketsProps {
  isPlaying: boolean;
  speed: number;
  onCollisionCoreReached: () => void;
}

// 3D Instanced Proton Beam Packets Mesh
const BeamPackets3D: React.FC<BeamPacketsProps> = ({ isPlaying, speed, onCollisionCoreReached }) => {
  const meshRef1 = useRef<THREE.InstancedMesh>(null);
  const meshRef2 = useRef<THREE.InstancedMesh>(null);
  const zPos1 = useRef<number>(-35);
  const zPos2 = useRef<number>(35);
  const hasTriggeredRef = useRef<boolean>(false);

  const particleCount = 200;
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Seeded deterministic offset positions
  const offsets = useMemo(() => {
    const prng = createDeterministicPRNG(4264);
    const arr: [number, number, number][] = [];
    for (let i = 0; i < particleCount; i++) {
      const radius = prng() * 0.4;
      const angle = prng() * Math.PI * 2;
      const zOffset = (prng() - 0.5) * 4;
      arr.push([Math.cos(angle) * radius, Math.sin(angle) * radius, zOffset]);
    }
    return arr;
  }, []);

  useFrame((_, delta) => {
    if (!isPlaying) return;

    const moveStep = delta * 15 * speed;

    // Beam 1 moves in +Z towards origin
    if (zPos1.current < 0) {
      zPos1.current += moveStep;
      if (zPos1.current > 0) zPos1.current = 0;
    }

    // Beam 2 moves in -Z towards origin
    if (zPos2.current > 0) {
      zPos2.current -= moveStep;
      if (zPos2.current < 0) zPos2.current = 0;
    }

    // Check collision core auto-pause trigger
    if (Math.abs(zPos1.current) < 0.5 && Math.abs(zPos2.current) < 0.5 && !hasTriggeredRef.current) {
      hasTriggeredRef.current = true;
      onCollisionCoreReached();
    }

    // Update instance matrix matrices without allocations
    if (meshRef1.current) {
      offsets.forEach(([ox, oy, oz], i) => {
        dummy.position.set(ox, oy, zPos1.current + oz);
        dummy.scale.setScalar(0.08);
        dummy.updateMatrix();
        meshRef1.current!.setMatrixAt(i, dummy.matrix);
      });
      meshRef1.current.instanceMatrix.needsUpdate = true;
    }

    if (meshRef2.current) {
      offsets.forEach(([ox, oy, oz], i) => {
        dummy.position.set(ox, oy, zPos2.current + oz);
        dummy.scale.setScalar(0.08);
        dummy.updateMatrix();
        meshRef2.current!.setMatrixAt(i, dummy.matrix);
      });
      meshRef2.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      {/* Beam 1 (+Z Direction, Cyan) */}
      <instancedMesh ref={meshRef1} args={[undefined, undefined, particleCount]}>
        <sphereGeometry args={[0.5, 12, 12]} />
        <meshStandardMaterial color="#06b6d4" emissive="#06b6d4" emissiveIntensity={0.8} />
      </instancedMesh>

      {/* Beam 2 (-Z Direction, Detector White) */}
      <instancedMesh ref={meshRef2} args={[undefined, undefined, particleCount]}>
        <sphereGeometry args={[0.5, 12, 12]} />
        <meshStandardMaterial color="#f8fafc" emissive="#f8fafc" emissiveIntensity={0.8} />
      </instancedMesh>

      {/* Interaction Core Glow Flash when converged */}
      {Math.abs(zPos1.current) < 1.0 && (
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[1.5, 32, 32]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.6} />
        </mesh>
      )}
    </group>
  );
};

// 3D Detector Chamber & Beam Tube Component
const AcceleratorScene3D: React.FC<{
  cameraPreset: CameraPreset;
  isPlaying: boolean;
  speed: number;
  onCollisionCoreReached: () => void;
}> = ({ cameraPreset, isPlaying, speed, onCollisionCoreReached }) => {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);

  // Position camera based on selected preset
  useEffect(() => {
    if (!cameraRef.current) return;
    const cam = cameraRef.current;
    switch (cameraPreset) {
      case 'tube':
        cam.position.set(0, 8, 30);
        cam.lookAt(0, 0, 0);
        break;
      case 'observatory':
        cam.position.set(22, 16, 22);
        cam.lookAt(0, 0, 0);
        break;
      case 'collision':
        cam.position.set(0, 2, 8);
        cam.lookAt(0, 0, 0);
        break;
      case 'longitudinal':
        cam.position.set(0, 0.5, 38);
        cam.lookAt(0, 0, 0);
        break;
      case 'free':
        break;
    }
  }, [cameraPreset]);

  return (
    <>
      <PerspectiveCamera ref={cameraRef} makeDefault position={[22, 16, 22]} fov={50} />
      <OrbitControls enablePan enableZoom enableRotate maxDistance={80} minDistance={3} />

      {/* Lighting Atmosphere */}
      <ambientLight intensity={0.4} />
      <pointLight position={[0, 10, 0]} intensity={1.2} color="#38bdf8" />
      <directionalLight position={[10, 20, 15]} intensity={0.8} />

      {/* Z-Axis Transparent Beam Tube */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <cylinderGeometry args={[1.2, 1.2, 70, 32, 1, true]} />
        <meshStandardMaterial color="#1e293b" transparent opacity={0.25} wireframe />
      </mesh>

      {/* Concentric Glass Detector Chamber Rings (Not to scale) */}
      <group position={[0, 0, 0]}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[4.5, 4.5, 12, 32, 1, true]} />
          <meshStandardMaterial color="#06b6d4" transparent opacity={0.15} wireframe />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[8.0, 8.0, 18, 32, 1, true]} />
          <meshStandardMaterial color="#f59e0b" transparent opacity={0.12} wireframe />
        </mesh>
      </group>

      {/* Beam Packets */}
      <BeamPackets3D isPlaying={isPlaying} speed={speed} onCollisionCoreReached={onCollisionCoreReached} />
    </>
  );
};

interface AcceleratorJourneyViewProps {
  onEnterStudio?: () => void;
}

export const AcceleratorJourneyView: React.FC<AcceleratorJourneyViewProps> = ({ onEnterStudio }) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1.0);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('observatory');
  const [collisionReached, setCollisionReached] = useState<boolean>(false);
  const [webglAvailable, setWebglAvailable] = useState<boolean>(true);

  // Test WebGL context availability on mount
  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) setWebglAvailable(false);
    } catch {
      setWebglAvailable(false);
    }
  }, []);

  const handleRestart = () => {
    setCollisionReached(false);
    setIsPlaying(true);
  };

  const handleCollisionCoreReached = () => {
    setIsPlaying(false);
    setCollisionReached(true);
  };

  if (!webglAvailable) {
    return (
      <div className="flex flex-col gap-6">
        <NonWebGLPolarView onEnterStudio={onEnterStudio} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      {/* 3D Observatory Visual Canvas Frame */}
      <div className="relative w-full h-[520px] bg-[#05070c] rounded-2xl border border-slate-800/80 overflow-hidden shadow-2xl">
        <Canvas>
          <AcceleratorScene3D
            cameraPreset={cameraPreset}
            isPlaying={isPlaying}
            speed={speed}
            onCollisionCoreReached={handleCollisionCoreReached}
          />
        </Canvas>

        {/* Floating Top Control Overlay */}
        <div className="absolute top-4 left-4 right-4 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
          {/* Telemetry Badge */}
          <div className="px-3 py-1.5 rounded-xl bg-[#090d16]/90 border border-slate-800 text-xs font-mono text-cyan-300 flex items-center gap-2 backdrop-blur-md pointer-events-auto">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span>Beam Energy: 4 TeV + 4 TeV (&radic;s = 8 TeV)</span>
          </div>

          {/* Camera Viewport Presets */}
          <div className="flex items-center gap-1 bg-[#090d16]/90 p-1 rounded-xl border border-slate-800 backdrop-blur-md pointer-events-auto">
            <span className="px-2 text-[10px] font-mono text-slate-400 uppercase flex items-center gap-1">
              <Camera className="w-3 h-3 text-cyan-400" /> Camera:
            </span>
            <button
              onClick={() => setCameraPreset('tube')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                cameraPreset === 'tube' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:text-white'
              }`}
            >
              1. Tube Profile
            </button>
            <button
              onClick={() => setCameraPreset('observatory')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                cameraPreset === 'observatory' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:text-white'
              }`}
            >
              2. 45&deg; Observatory
            </button>
            <button
              onClick={() => setCameraPreset('collision')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                cameraPreset === 'collision' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:text-white'
              }`}
            >
              3. Collision Core
            </button>
            <button
              onClick={() => setCameraPreset('longitudinal')}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                cameraPreset === 'longitudinal' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:text-white'
              }`}
            >
              4. Longitudinal
            </button>
          </div>
        </div>

        {/* Collision Core Portal Overlay Action */}
        {collisionReached && (
          <div className="absolute inset-0 bg-[#05070c]/70 backdrop-blur-sm flex flex-col items-center justify-center gap-4 text-center p-6 z-30 animate-fadeIn">
            <div className="p-3 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-300">
              <Activity className="w-8 h-8 animate-pulse" />
            </div>
            <h3 className="text-xl font-bold text-white">Beam Convergence Achieved at Interaction Core</h3>
            <p className="text-sm text-slate-300 max-w-lg">
              Proton beam packets have reached the interaction point. You can now step into the Event Reconstruction Studio to inspect real recorded ATLAS open-data event kinematics.
            </p>
            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={handleRestart}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs transition-colors flex items-center gap-2 border border-slate-700"
              >
                <RotateCcw className="w-4 h-4" /> Replay Journey
              </button>
              {onEnterStudio && (
                <button
                  onClick={onEnterStudio}
                  className="px-6 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/30 transition-all flex items-center gap-2"
                >
                  Enter Event Reconstruction Studio <ArrowRight className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        )}

        {/* Floating Bottom Controls */}
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between gap-3 pointer-events-none">
          <div className="flex items-center gap-2 bg-[#090d16]/90 p-1.5 rounded-xl border border-slate-800 backdrop-blur-md pointer-events-auto">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 transition-colors"
              aria-label={isPlaying ? 'Pause beam animation' : 'Start beam animation'}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={handleRestart}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              aria-label="Restart beam animation"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <div className="h-4 w-px bg-slate-800 mx-1" />
            <div className="flex items-center gap-1">
              <Gauge className="w-3.5 h-3.5 text-slate-400 ml-1" />
              {[0.5, 1.0, 2.0].map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                    speed === s ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-[#090d16]/90 border border-slate-800 text-[11px] font-mono text-slate-400 hidden sm:block pointer-events-auto">
            Seed: 4264 &bull; Deterministic Proton Packets
          </div>
        </div>
      </div>
    </div>
  );
};
