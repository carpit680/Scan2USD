import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

export interface SparseCloudPickerProps {
  plyUrl: string;
  floorMatrix: number[][];
  onEdgeChange: (points: [THREE.Vector3, THREE.Vector3] | null, sourceLength: number | null) => void;
}

function mat4FromRows(rows: number[][]): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(
    rows[0][0],
    rows[0][1],
    rows[0][2],
    rows[0][3],
    rows[1][0],
    rows[1][1],
    rows[1][2],
    rows[1][3],
    rows[2][0],
    rows[2][1],
    rows[2][2],
    rows[2][3],
    rows[3][0],
    rows[3][1],
    rows[3][2],
    rows[3][3],
  );
  return m;
}

export function SparseCloudPicker({ plyUrl, floorMatrix, onEdgeChange }: SparseCloudPickerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const onEdgeChangeRef = useRef(onEdgeChange);
  onEdgeChangeRef.current = onEdgeChange;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 640;
    const height = Math.max(360, mount.clientHeight || 480);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f14);

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 5000);
    camera.position.set(2, -4, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.45);
    dir.position.set(3, 5, 4);
    scene.add(dir);
    scene.add(new THREE.AxesHelper(0.5));

    const floorT = mat4FromRows(floorMatrix);
    let pointsObj: THREE.Points | null = null;
    let disposed = false;
    let markerRadius = 0.03;

    const pickMarkers: THREE.Mesh[] = [];
    const pickPositions: THREE.Vector3[] = [];
    let edgeLine: THREE.Line | null = null;

    const markerGeo = new THREE.SphereGeometry(1, 12, 12);
    const markerMatA = new THREE.MeshBasicMaterial({ color: 0x3dd6c6 });
    const markerMatB = new THREE.MeshBasicMaterial({ color: 0xf07178 });

    function clearPicks() {
      for (const m of pickMarkers) {
        scene.remove(m);
      }
      pickMarkers.length = 0;
      pickPositions.length = 0;
      if (edgeLine) {
        scene.remove(edgeLine);
        edgeLine.geometry.dispose();
        (edgeLine.material as THREE.Material).dispose();
        edgeLine = null;
      }
      onEdgeChangeRef.current(null, null);
    }

    function updateEdgeVisual() {
      if (edgeLine) {
        scene.remove(edgeLine);
        edgeLine.geometry.dispose();
        (edgeLine.material as THREE.Material).dispose();
        edgeLine = null;
      }
      if (pickPositions.length === 2) {
        const geo = new THREE.BufferGeometry().setFromPoints(pickPositions);
        edgeLine = new THREE.Line(
          geo,
          new THREE.LineBasicMaterial({ color: 0xffcc66 }),
        );
        scene.add(edgeLine);
        const dist = pickPositions[0].distanceTo(pickPositions[1]);
        onEdgeChangeRef.current(
          [pickPositions[0].clone(), pickPositions[1].clone()],
          dist,
        );
      } else {
        onEdgeChangeRef.current(null, null);
      }
    }

    function addPick(world: THREE.Vector3) {
      if (pickPositions.length >= 2) {
        clearPicks();
      }
      const mat = pickPositions.length === 0 ? markerMatA : markerMatB;
      const mesh = new THREE.Mesh(markerGeo, mat);
      mesh.position.copy(world);
      mesh.scale.setScalar(markerRadius);
      scene.add(mesh);
      pickMarkers.push(mesh);
      pickPositions.push(world.clone());
      updateEdgeVisual();
    }

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.05 };
    const pointer = new THREE.Vector2();
    let downX = 0;
    let downY = 0;

    function onPointerDownTrack(event: PointerEvent) {
      downX = event.clientX;
      downY = event.clientY;
    }

    function onPointerUp(event: PointerEvent) {
      if (!pointsObj || event.button !== 0) return;
      const dx = event.clientX - downX;
      const dy = event.clientY - downY;
      if (dx * dx + dy * dy > 16) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(pointsObj, false);
      if (hits.length === 0) return;
      addPick(hits[0].point);
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDownTrack);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    const loader = new PLYLoader();
    loader.load(
      plyUrl,
      (geometry) => {
        if (disposed) {
          geometry.dispose();
          return;
        }
        const pos = geometry.getAttribute("position");
        const v = new THREE.Vector3();
        for (let i = 0; i < pos.count; i++) {
          v.fromBufferAttribute(pos, i);
          v.applyMatrix4(floorT);
          pos.setXYZ(i, v.x, v.y, v.z);
        }
        pos.needsUpdate = true;
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        const hasColor = Boolean(geometry.getAttribute("color"));
        const material = new THREE.PointsMaterial({
          size: 0.025,
          sizeAttenuation: true,
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0x8ab4f8,
        });
        pointsObj = new THREE.Points(geometry, material);
        scene.add(pointsObj);

        const box = geometry.boundingBox;
        if (box) {
          const center = new THREE.Vector3();
          box.getCenter(center);
          const size = new THREE.Vector3();
          box.getSize(size);
          const radius = Math.max(size.x, size.y, size.z, 0.5);
          markerRadius = Math.max(0.015, radius * 0.006);
          controls.target.copy(center);
          camera.position
            .copy(center)
            .add(new THREE.Vector3(radius * 0.8, -radius * 1.2, radius * 0.9));
          camera.near = Math.max(0.001, radius / 500);
          camera.far = radius * 50;
          camera.updateProjectionMatrix();
          controls.update();
          raycaster.params.Points = { threshold: Math.max(0.02, radius * 0.008) };
          material.size = Math.max(0.01, radius * 0.004);
        }
      },
      undefined,
      (err) => {
        console.error("Failed to load sparse_pc.ply", err);
      },
    );

    let frame = 0;
    function animate() {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    const ro = new ResizeObserver(() => {
      if (!mount) return;
      const w = mount.clientWidth || width;
      const h = Math.max(360, mount.clientHeight || height);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(mount);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDownTrack);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      clearPicks();
      controls.dispose();
      renderer.dispose();
      if (pointsObj) {
        pointsObj.geometry.dispose();
        (pointsObj.material as THREE.Material).dispose();
      }
      markerGeo.dispose();
      markerMatA.dispose();
      markerMatB.dispose();
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [plyUrl, floorMatrix]);

  return (
    <div
      ref={mountRef}
      className="h-[min(70vh,640px)] w-full overflow-hidden rounded-xl border border-ink-700 bg-ink-950"
      title="Orbit: drag · Pick edge: click two points"
    />
  );
}
