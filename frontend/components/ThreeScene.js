import { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import { Box, Spinner } from '@chakra-ui/react';

// A smaller or simpler “network.”
const NETWORK_LAYERS = [
  { label: 'Data', count: 4, x: -4 },
  { label: 'ML/AI', count: 6, x: 0 },
  { label: 'Outcome', count: 3, x: 4 },
];

function Node({ position }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.25, 16, 16]} />
      <meshStandardMaterial
        color="#00FF88"
        emissive="#00FF88"
        emissiveIntensity={0.3}
        metalness={0.5}
        roughness={0.4}
      />
    </mesh>
  );
}

function Network() {
  const layersNodes = useMemo(() => {
    return NETWORK_LAYERS.map((layer) => {
      const positions = [];
      const spacing = 1.4;
      const startY = -((layer.count - 1) * spacing) / 2;
      for (let i = 0; i < layer.count; i++) {
        positions.push([layer.x, startY + i * spacing, 0]);
      }
      return { label: layer.label, positions };
    });
  }, []);

  return (
    <group>
      {layersNodes.map((layer, idx) => (
        <group key={layer.label}>
          <Text
            position={[NETWORK_LAYERS[idx].x, 4.5, 0]}
            fontSize={0.5}
            color="white"
            anchorX="center"
          >
            {layer.label}
          </Text>
          {layer.positions.map((pos, i) => (
            <Node key={`node-${idx}-${i}`} position={pos} />
          ))}
        </group>
      ))}

      {/* Connect layers with lines */}
      {layersNodes.slice(0, -1).map((_, i) =>
        layersNodes[i].positions.map((pos1) =>
          layersNodes[i + 1].positions.map((pos2) => (
            <Line
              key={`line-${i}-${pos1}-${pos2}`}
              points={[pos1, pos2]}
              color="white"
              lineWidth={0.4}
              opacity={0.7}
              transparent
            />
          ))
        )
      )}
    </group>
  );
}

export default function ThreeScene() {
  const [isClient, setIsClient] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setIsClient(true);
    const timer = setTimeout(() => setLoading(false), 500);
    return () => clearTimeout(timer);
  }, []);

  if (!isClient) {
    return (
      <Box
        bg="gray.900"
        h="100%"
        w="100%"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Spinner size="xl" color="#00FF88" />
      </Box>
    );
  }

  if (loading) {
    return (
      <Box
        bg="gray.900"
        h="100%"
        w="100%"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Spinner size="xl" color="#00FF88" />
      </Box>
    );
  }

  return (
    <Canvas
      camera={{ position: [0, 0, 12], fov: 45 }}
      style={{ background: '#141414', width: '100%', height: '100%' }}
    >
      <ambientLight intensity={0.25} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#00FF88" />
      <group scale={[0.75, 0.75, 0.75]}>
        <Network />
      </group>
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.3}
        enableZoom={false}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 3}
      />
    </Canvas>
  );
}
