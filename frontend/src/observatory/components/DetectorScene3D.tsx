import React, { useRef, useMemo, useEffect, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Html } from '@react-three/drei';
import * as THREE from 'three';
import { ReconstructedObject } from '../../types';
import { etaPhiPtToCartesian } from '../../lib/kinematics';
import { createDeterministicPRNG } from '../../lib/journeyState';

export type SpeedMode = 'L_SPEED' | 'VIEWABLE';
export type SpeedMultiplier = 0.2 | 0.5 | 1 | 2;

export interface IllustrativeParticle {
  id: number;
  dirX: number;
  dirY: number;
  dirZ: number;
  speed: number;
  maxRadius: number;
  color: string;
  typeLabel: string;
}

export interface SelectedObjectPayload {
  isApiBacked: boolean;
  realObject?: ReconstructedObject;
  illustrativeParticle?: IllustrativeParticle;
  cartesianPxPyPz?: { px: number; py: number; pz: number };
}

interface DetectorScene3DProps {
  isPlaying: boolean;
  speedMode: SpeedMode;
  speedMultiplier: SpeedMultiplier;
  zoomLevel: number; // Controlled zoom distance target (e.g. 5 to 50)
  eventId: number;
  realObjects: ReconstructedObject[];
  metData?: { magnitude: number; phi: number } | null;
  onSelectObject: (payload: SelectedObjectPayload) => void;
  onAnimationPhaseChange?: (phase: 'BEAM' | 'FLASH' | 'BURST' | 'STATIC') => void;
}

// Color palette definitions according to requirements
const COLOR_TAU = '#06b6d4'; // Cyan
const COLOR_LEPTON = '#f8fafc'; // White / Pale Blue
const COLOR_JET_LEADING = '#f59e0b'; // Gold / Orange
const COLOR_JET_SUBLEADING = '#fbbf24'; // Gold / Amber
const COLOR_MET = '#ec4899'; // Magenta
const COLOR_BEAM_1 = '#06b6d4'; // Cyan (+Z)
const COLOR_BEAM_2 = '#f43f5e'; // Red/Magenta (-Z)

/**
 * 3D Detector Viewport Scene with beam packets, collision flash,
 * deterministic illustrative burst, real physics vectors, and interactive camera controls.
 */
export const DetectorScene3D: React.FC<DetectorScene3DProps> = ({
  isPlaying,
  speedMode,
  speedMultiplier,
  zoomLevel,
  eventId,
  realObjects,
  metData,
  onSelectObject,
  onAnimationPhaseChange,
}) => {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  // Animation timeline state references (zero per-frame allocations)
  const animTimeRef = useRef<number>(0);
  const phaseRef = useRef<'BEAM' | 'FLASH' | 'BURST' | 'STATIC'>('BEAM');

  // Beam 1 (+Z) and Beam 2 (-Z) packet refs
  const beam1GroupRef = useRef<THREE.Group>(null);
  const beam2GroupRef = useRef<THREE.Group>(null);
  const flashMeshRef = useRef<THREE.Mesh>(null);
  const flashMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const burstGroupRef = useRef<THREE.Group>(null);

  // Pre-allocated dummy objects for performance (no allocations inside useFrame)
  const dummyObj = useMemo(() => new THREE.Object3D(), []);

  // Generate deterministic illustrative burst particles based on event ID seed
  const illustrativeParticles = useMemo<IllustrativeParticle[]>(() => {
    const prng = createDeterministicPRNG(eventId || 4264);
    const particles: IllustrativeParticle[] = [];
    const count = 90;
    const colors = [COLOR_TAU, COLOR_LEPTON, COLOR_JET_LEADING, COLOR_MET, '#38bdf8', '#a855f7'];
    const types = [
      'Illustrative Hadron Track',
      'Illustrative Electromagnetic Cluster',
      'Illustrative Muon Track',
      'Illustrative Neutral Calorimeter Jet',
      'Illustrative Secondary Particle Track',
    ];

    for (let i = 0; i < count; i++) {
      // Random direction on unit sphere
      const theta = prng() * Math.PI * 2;
      const phi = Math.acos(2 * prng() - 1);
      const dirX = Math.sin(phi) * Math.cos(theta);
      const dirY = Math.sin(phi) * Math.sin(theta);
      const dirZ = Math.cos(phi);

      particles.push({
        id: i,
        dirX,
        dirY,
        dirZ,
        speed: 0.8 + prng() * 1.4,
        maxRadius: 10 + prng() * 14,
        color: colors[Math.floor(prng() * colors.length)],
        typeLabel: types[Math.floor(prng() * types.length)],
      });
    }
    return particles;
  }, [eventId]);

  // Seeded beam packet particle offsets
  const beamOffsets = useMemo(() => {
    const prng = createDeterministicPRNG(1234 + (eventId % 100));
    const offsets: [number, number, number][] = [];
    for (let i = 0; i < 40; i++) {
      const radius = prng() * 0.35;
      const angle = prng() * Math.PI * 2;
      const zOff = (prng() - 0.5) * 3;
      offsets.push([Math.cos(angle) * radius, Math.sin(angle) * radius, zOff]);
    }
    return offsets;
  }, [eventId]);

  // Update camera position smoothly when zoomLevel changes
  useEffect(() => {
    if (!camera) return;
    const currentPos = camera.position.clone();
    const dir = currentPos.clone().normalize();
    const targetDist = Math.max(5, Math.min(50, zoomLevel));
    camera.position.copy(dir.multiplyScalar(targetDist));
    if (controlsRef.current) {
      controlsRef.current.update();
    }
  }, [zoomLevel, camera]);

  // Check prefers-reduced-motion
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  // Reset animation when eventId changes or restart requested
  const resetAnimation = useCallback(() => {
    animTimeRef.current = 0;
    phaseRef.current = 'BEAM';
    if (onAnimationPhaseChange) onAnimationPhaseChange('BEAM');
  }, [onAnimationPhaseChange]);

  useEffect(() => {
    resetAnimation();
  }, [eventId, resetAnimation]);

  // Main frame loop (No state updates, no per-frame allocations)
  useFrame((_, delta) => {
    if (prefersReducedMotion) return;

    if (speedMode === 'L_SPEED') {
      // Near-instant beam packets, collision flash, skipped burst
      if (beam1GroupRef.current) beam1GroupRef.current.position.z = 0;
      if (beam2GroupRef.current) beam2GroupRef.current.position.z = 0;
      if (flashMeshRef.current) flashMeshRef.current.scale.set(0.1, 0.1, 0.1);
      if (flashMatRef.current) flashMatRef.current.opacity = 0;
      if (burstGroupRef.current) burstGroupRef.current.visible = false;
      if (phaseRef.current !== 'STATIC') {
        phaseRef.current = 'STATIC';
        if (onAnimationPhaseChange) onAnimationPhaseChange('STATIC');
      }
      return;
    }

    if (burstGroupRef.current) burstGroupRef.current.visible = true;

    if (!isPlaying) return;

    const timeStep = delta * speedMultiplier;
    animTimeRef.current += timeStep;

    const totalDuration = 6.0;
    const currentLoopTime = animTimeRef.current % totalDuration;

    // Timeline breakdown:
    // 0s to 2.2s: Beam approach
    // 2.2s to 3.0s: Flash
    // 3.0s to 6.0s: Burst expansion
    if (currentLoopTime < 2.2) {
      if (phaseRef.current !== 'BEAM') {
        phaseRef.current = 'BEAM';
        if (onAnimationPhaseChange) onAnimationPhaseChange('BEAM');
      }
      const progress = currentLoopTime / 2.2; // 0 to 1
      const z1 = -35 * (1 - progress);
      const z2 = 35 * (1 - progress);

      if (beam1GroupRef.current) beam1GroupRef.current.position.z = z1;
      if (beam2GroupRef.current) beam2GroupRef.current.position.z = z2;
      if (flashMeshRef.current) flashMeshRef.current.scale.set(0.01, 0.01, 0.01);
      if (flashMatRef.current) flashMatRef.current.opacity = 0;
      if (burstGroupRef.current) burstGroupRef.current.children.forEach((c) => (c.visible = false));
    } else if (currentLoopTime >= 2.2 && currentLoopTime < 3.0) {
      if (phaseRef.current !== 'FLASH') {
        phaseRef.current = 'FLASH';
        if (onAnimationPhaseChange) onAnimationPhaseChange('FLASH');
      }
      if (beam1GroupRef.current) beam1GroupRef.current.position.z = 0;
      if (beam2GroupRef.current) beam2GroupRef.current.position.z = 0;

      const flashProgress = (currentLoopTime - 2.2) / 0.8;
      const flashScale = 1 + flashProgress * 3.5;
      const opacity = Math.max(0, 1 - flashProgress);

      if (flashMeshRef.current) flashMeshRef.current.scale.setScalar(flashScale);
      if (flashMatRef.current) flashMatRef.current.opacity = opacity;
    } else {
      if (phaseRef.current !== 'BURST') {
        phaseRef.current = 'BURST';
        if (onAnimationPhaseChange) onAnimationPhaseChange('BURST');
      }
      const burstProgress = (currentLoopTime - 3.0) / 3.0; // 0 to 1

      if (flashMatRef.current) flashMatRef.current.opacity = 0;

      if (burstGroupRef.current) {
        burstGroupRef.current.children.forEach((child, idx) => {
          child.visible = true;
          const p = illustrativeParticles[idx];
          if (p) {
            const currentDist = p.maxRadius * burstProgress * p.speed;
            child.position.set(p.dirX * currentDist, p.dirY * currentDist, p.dirZ * currentDist);
          }
        });
      }
    }
  });

  return (
    <>
      <PerspectiveCamera makeDefault position={[22, 16, 22]} fov={45} />
      <OrbitControls
        ref={controlsRef}
        enablePan
        enableZoom
        enableRotate
        minDistance={4}
        maxDistance={50}
      />

      {/* Ambient & Directional Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[0, 0, 0]} intensity={1.8} color="#38bdf8" />
      <directionalLight position={[10, 20, 15]} intensity={0.8} color="#f8fafc" />

      {/* Concentric Detector Shell Geometry (Low-opacity cyan/slate wireframes) */}
      <group position={[0, 0, 0]}>
        {/* Inner Pixel Tracker */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[3.5, 3.5, 10, 32, 1, true]} />
          <meshStandardMaterial color="#06b6d4" transparent opacity={0.12} wireframe />
        </mesh>
        {/* Electromagnetic Calorimeter (ECAL) */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[7.0, 7.0, 14, 32, 1, true]} />
          <meshStandardMaterial color="#38bdf8" transparent opacity={0.1} wireframe />
        </mesh>
        {/* Hadronic Calorimeter (HCAL) */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[11.5, 11.5, 18, 32, 1, true]} />
          <meshStandardMaterial color="#f59e0b" transparent opacity={0.08} wireframe />
        </mesh>
        {/* Outer Muon Spectrometer Rings */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[16.0, 16.0, 24, 32, 1, true]} />
          <meshStandardMaterial color="#64748b" transparent opacity={0.15} wireframe />
        </mesh>
      </group>

      {/* Transparent Z-Axis Beam Tube */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <cylinderGeometry args={[1.1, 1.1, 70, 32, 1, true]} />
        <meshStandardMaterial color="#0891b2" transparent opacity={0.25} wireframe />
      </mesh>

      {/* Proton Beam 1 Packet (+Z Direction, Cyan) */}
      <group ref={beam1GroupRef} position={[0, 0, -35]}>
        {beamOffsets.map(([ox, oy, oz], i) => (
          <mesh key={`b1-${i}`} position={[ox, oy, oz]}>
            <sphereGeometry args={[0.12, 8, 8]} />
            <meshBasicMaterial color={COLOR_BEAM_1} />
          </mesh>
        ))}
      </group>

      {/* Proton Beam 2 Packet (-Z Direction, Red/Magenta) */}
      <group ref={beam2GroupRef} position={[0, 0, 35]}>
        {beamOffsets.map(([ox, oy, oz], i) => (
          <mesh key={`b2-${i}`} position={[ox, oy, oz]}>
            <sphereGeometry args={[0.12, 8, 8]} />
            <meshBasicMaterial color={COLOR_BEAM_2} />
          </mesh>
        ))}
      </group>

      {/* Collision Central Interaction Core Flash Mesh */}
      <mesh ref={flashMeshRef} position={[0, 0, 0]}>
        <sphereGeometry args={[1.8, 32, 32]} />
        <meshBasicMaterial ref={flashMatRef} color="#38bdf8" transparent opacity={0} />
      </mesh>

      {/* Illustrative Educational Burst Particles Group */}
      <group ref={burstGroupRef}>
        {illustrativeParticles.map((p) => (
          <mesh
            key={`ill-${p.id}`}
            position={[0, 0, 0]}
            onClick={(e) => {
              e.stopPropagation();
              onSelectObject({
                isApiBacked: false,
                illustrativeParticle: p,
              });
            }}
          >
            <sphereGeometry args={[0.16, 8, 8]} />
            <meshBasicMaterial color={p.color} />
          </mesh>
        ))}
      </group>

      {/* Overlay API-Backed Real Reconstructed Event Vectors */}
      {realObjects.map((obj, index) => {
        const cart = etaPhiPtToCartesian(obj.pt, obj.eta, obj.phi);
        const scale = 0.1;
        const target = new THREE.Vector3(cart.x * scale, cart.y * scale, cart.z * scale);
        const length = target.length();

        let color = COLOR_TAU;
        if (obj.object_type === 'lepton') color = COLOR_LEPTON;
        if (obj.object_type === 'jet_leading') color = COLOR_JET_LEADING;
        if (obj.object_type === 'jet_subleading') color = COLOR_JET_SUBLEADING;

        return (
          <group
            key={`real-obj-${obj.object_type}-${index}`}
            onClick={(e) => {
              e.stopPropagation();
              onSelectObject({
                isApiBacked: true,
                realObject: obj,
                cartesianPxPyPz: { px: cart.x, py: cart.y, pz: cart.z },
              });
            }}
          >
            {/* Cone Vector Ray */}
            <mesh position={[target.x / 2, target.y / 2, target.z / 2]}>
              <cylinderGeometry args={[0.06, 0.35, length, 16]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} />
            </mesh>

            {/* Glowing Vector Tip */}
            <mesh position={[target.x, target.y, target.z]}>
              <sphereGeometry args={[0.3, 16, 16]} />
              <meshBasicMaterial color={color} />
            </mesh>

            {/* Floating 3D Text Label Overlay */}
            <Html position={[target.x * 1.15, target.y * 1.15, target.z * 1.15]} center>
              <div className="px-2 py-0.5 rounded bg-[#090d16]/90 border border-slate-700 text-[10px] font-mono text-white whitespace-nowrap shadow-md cursor-pointer hover:border-cyan-400">
                {obj.label}
              </div>
            </Html>
          </group>
        );
      })}

      {/* MET Vector Ray (Dashed Magenta) */}
      {metData && metData.magnitude > 0 && (() => {
        const metScale = 0.1;
        const mx = metData.magnitude * Math.cos(metData.phi) * metScale;
        const my = metData.magnitude * Math.sin(metData.phi) * metScale;

        return (
          <group
            onClick={(e) => {
              e.stopPropagation();
              onSelectObject({
                isApiBacked: true,
                realObject: {
                  object_type: 'met',
                  label: 'MET / missing transverse energy',
                  pt: metData.magnitude,
                  eta: 0,
                  phi: metData.phi,
                  present: true,
                },
                cartesianPxPyPz: {
                  px: metData.magnitude * Math.cos(metData.phi),
                  py: metData.magnitude * Math.sin(metData.phi),
                  pz: 0,
                },
              });
            }}
          >
            <mesh position={[mx / 2, my / 2, 0]}>
              <boxGeometry args={[Math.sqrt(mx * mx + my * my), 0.08, 0.08]} />
              <meshBasicMaterial color={COLOR_MET} />
            </mesh>
            <mesh position={[mx, my, 0]}>
              <sphereGeometry args={[0.25, 12, 12]} />
              <meshBasicMaterial color={COLOR_MET} />
            </mesh>
            <Html position={[mx * 1.1, my * 1.1, 0]} center>
              <div className="px-2 py-0.5 rounded bg-pink-950/90 border border-pink-500/50 text-[10px] font-mono text-pink-300 whitespace-nowrap cursor-pointer">
                MET: {metData.magnitude.toFixed(1)} GeV
              </div>
            </Html>
          </group>
        );
      })()}
    </>
  );
};
