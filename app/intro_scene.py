"""
Loop Guard - 3D Closed-Loop Animation Scene
Renders a lightweight, interactive-free Three.js animation of a PID control loop.
Features:
- Pinned Three.js r128 from cdnjs
- 5 labeled nodes: Setpoint -> Controller -> Valve -> Process -> Sensor
- High-res crisp label sprites with dark rounded pill backgrounds
- Glowing travel pulses and periodic fault detection flicker
- Capped pixel ratio (2x), paused when tab is hidden, prefers-reduced-motion support
- Responsive camera fitting bounding sphere tightly (~6% margin) for full frame fill
- Instant static fallback diagram if WebGL/CDN is unavailable
"""

def build_intro_scene_html() -> str:
    """
    Build HTML/JS string for the 3D loop animation component.
    All lines are stripped to guarantee no leading spaces or blank lines.
    """
    raw_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
width: 100%;
height: 100%;
margin: 0;
padding: 0;
overflow: hidden;
background: transparent;
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#webgl-container {
width: 100%;
height: 100%;
margin: 0;
padding: 0;
overflow: hidden;
position: absolute;
top: 0;
left: 0;
}
canvas {
display: block;
width: 100%;
height: 100%;
margin: 0;
overflow: hidden;
}
#fallback {
display: none;
width: 100%;
height: 100%;
padding: 28px 20px;
background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0d1117 100%);
border-radius: 12px;
border: 1px solid rgba(255, 255, 255, 0.08);
flex-direction: column;
justify-content: center;
align-items: center;
text-align: center;
}
.fb-title {
font-size: 1.15rem;
font-weight: 700;
color: #f8fafc;
margin-bottom: 24px;
letter-spacing: 0.02em;
}
.fb-flow {
display: flex;
align-items: center;
justify-content: center;
gap: 10px;
flex-wrap: wrap;
max-width: 780px;
}
.fb-node {
background: rgba(56, 189, 248, 0.12);
color: #38bdf8;
border: 1px solid rgba(56, 189, 248, 0.35);
padding: 8px 14px;
border-radius: 8px;
font-weight: 600;
font-size: 0.92rem;
}
.fb-arrow {
color: #94a3b8;
font-size: 1.1rem;
}
.fb-caption {
margin-top: 24px;
color: #94a3b8;
font-size: 0.85rem;
}
</style>
</head>
<body>
<div id="webgl-container"></div>
<div id="fallback">
<div class="fb-title">Closed Loop Feedback Architecture</div>
<div class="fb-flow">
<span class="fb-node">Setpoint</span>
<span class="fb-arrow">&rarr;</span>
<span class="fb-node">Controller</span>
<span class="fb-arrow">&rarr;</span>
<span class="fb-node">Valve</span>
<span class="fb-arrow">&rarr;</span>
<span class="fb-node">Process</span>
<span class="fb-arrow">&rarr;</span>
<span class="fb-node">Sensor</span>
<span class="fb-arrow">&rarr;</span>
<span class="fb-node">Feedback Error</span>
</div>
<div class="fb-caption">Continuous physical feedback loop passively inspected by Loop Guard.</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
function showFallback() {
var wc = document.getElementById("webgl-container");
var fb = document.getElementById("fallback");
if (wc) wc.style.display = "none";
if (fb) fb.style.display = "flex";
}
if (!window.WebGLRenderingContext || !window.THREE) {
showFallback();
return;
}
var container = document.getElementById("webgl-container");
var scene, camera, renderer;
var loopGroup;
var nodeMeshes = [];
var lineMeshes = [];
var pulseMeshes = [];
var nodePositions = [];
var radius = 2.7;
var nodeCount = 5;
var nodeNames = ["Setpoint", "Controller", "Valve", "Process", "Sensor"];
var nodeColors = [0x38bdf8, 0x38bdf8, 0x10b981, 0xf59e0b, 0x6366f1];
var isPaused = false;
var animId = null;

var spriteRadialFactor = 1.24;
var spriteHalfWidth = 1.05;
var spriteY = 0.35;
var spriteHalfHeight = 0.2625;
var spriteMaxR = Math.sqrt(Math.pow(radius * spriteRadialFactor + spriteHalfWidth, 2) + Math.pow(spriteY + spriteHalfHeight, 2));
var nodeSphereRadius = 0.18;
var nodeMaxR = radius + nodeSphereRadius;
var boundingRadius = Math.max(spriteMaxR, nodeMaxR);
var marginFactor = 1.06;
var effectiveRadius = boundingRadius * marginFactor;

var driftX = 0.25;
var driftY = 0.15;
var maxDrift = Math.sqrt(driftX * driftX + driftY * driftY);
var camPitch = Math.atan2(2.2, 5.8);
var camDistance = 12.5;
var camY0 = camDistance * Math.sin(camPitch);
var camZ0 = camDistance * Math.cos(camPitch);

function updateDimensions() {
if (!container || !camera || !renderer) return;
var width = container.clientWidth || window.innerWidth || 800;
var height = container.clientHeight || window.innerHeight || 500;
var aspect = width / height;
camera.aspect = aspect;
var vHalfTan = Math.tan((camera.fov * Math.PI) / 360);
var hHalfTan = aspect * vHalfTan;
var limitingTan = Math.min(vHalfTan, hHalfTan);
var limitingHalfAngle = Math.atan(limitingTan);
var baseDist = effectiveRadius / Math.sin(limitingHalfAngle);
camDistance = baseDist + maxDrift;
camY0 = camDistance * Math.sin(camPitch);
camZ0 = camDistance * Math.cos(camPitch);
camera.updateProjectionMatrix();
renderer.setSize(width, height);
}

try {
scene = new THREE.Scene();
camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
renderer.domElement.style.display = "block";
renderer.domElement.style.width = "100%";
renderer.domElement.style.height = "100%";
renderer.domElement.style.margin = "0";
renderer.domElement.style.overflow = "hidden";
container.appendChild(renderer.domElement);

updateDimensions();
camera.position.set(0, camY0, camZ0);
camera.lookAt(0, 0, 0);

loopGroup = new THREE.Group();
scene.add(loopGroup);

for (var i = 0; i < nodeCount; i++) {
var angle = (i / nodeCount) * Math.PI * 2 - Math.PI / 2;
var x = Math.cos(angle) * radius;
var z = Math.sin(angle) * radius;
nodePositions.push(new THREE.Vector3(x, 0, z));
var nodeGeo = new THREE.SphereGeometry(0.18, 16, 16);
var nodeMat = new THREE.MeshBasicMaterial({ color: nodeColors[i] });
var nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
nodeMesh.position.set(x, 0, z);
loopGroup.add(nodeMesh);
nodeMeshes.push(nodeMesh);
var canvas = document.createElement("canvas");
canvas.width = 512;
canvas.height = 128;
var ctx = canvas.getContext("2d");
ctx.clearRect(0, 0, 512, 128);
ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
ctx.roundRect ? ctx.roundRect(16, 16, 480, 96, 20) : ctx.rect(16, 16, 480, 96);
ctx.fill();
ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
ctx.lineWidth = 3;
ctx.stroke();
ctx.font = "bold 40px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
ctx.fillStyle = "#ffffff";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillText(nodeNames[i], 256, 64);
var texture = new THREE.CanvasTexture(canvas);
var spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
var sprite = new THREE.Sprite(spriteMat);
sprite.position.set(x * spriteRadialFactor, spriteY, z * spriteRadialFactor);
sprite.scale.set(2.1, 0.525, 1);
loopGroup.add(sprite);
}

for (var j = 0; j < nodeCount; j++) {
var p1 = nodePositions[j];
var p2 = nodePositions[(j + 1) % nodeCount];
var curve = new THREE.LineCurve3(p1, p2);
var tubeGeo = new THREE.TubeGeometry(curve, 16, 0.035, 8, false);
var tubeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.75 });
var tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
loopGroup.add(tubeMesh);
lineMeshes.push(tubeMesh);
var pGeo = new THREE.SphereGeometry(0.08, 10, 10);
var pMat = new THREE.MeshBasicMaterial({ color: 0x67e8f9 });
var pMesh = new THREE.Mesh(pGeo, pMat);
loopGroup.add(pMesh);
pulseMeshes.push(pMesh);
}

loopGroup.rotation.x = 0.42;

var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (prefersReduced) {
camera.position.set(0, camY0, camZ0);
camera.lookAt(0, 0, 0);
renderer.render(scene, camera);
return;
}

window.addEventListener("resize", updateDimensions);
window.addEventListener("orientationchange", updateDimensions);

document.addEventListener("visibilitychange", function() {
if (document.hidden) {
isPaused = true;
if (animId) cancelAnimationFrame(animId);
} else {
isPaused = false;
animId = requestAnimationFrame(animate);
}
});

var clock = new THREE.Clock();
function animate() {
if (isPaused) return;
animId = requestAnimationFrame(animate);
var elapsed = clock.getElapsedTime();
loopGroup.rotation.y = elapsed * 0.22;
loopGroup.rotation.x = 0.42 + Math.sin(elapsed * 0.35) * 0.05;
camera.position.x = Math.sin(elapsed * 0.2) * driftX;
camera.position.y = camY0 + Math.cos(elapsed * 0.25) * driftY;
camera.position.z = camZ0;
camera.lookAt(0, 0, 0);
for (var k = 0; k < nodeCount; k++) {
var segProgress = (elapsed * 0.35 + k * (1.0 / nodeCount)) % 1.0;
var segIdx = Math.floor(segProgress * nodeCount);
var localT = (segProgress * nodeCount) - segIdx;
var pA = nodePositions[segIdx];
var pB = nodePositions[(segIdx + 1) % nodeCount];
pulseMeshes[k].position.lerpVectors(pA, pB, localT);
}
var cycleTime = elapsed % 4.5;
var faultSegment = lineMeshes[2];
if (cycleTime > 2.0 && cycleTime < 2.9) {
var flicker = Math.sin(elapsed * 24) > 0;
faultSegment.material.color.setHex(flicker ? 0xef4444 : 0x38bdf8);
faultSegment.material.opacity = flicker ? 1.0 : 0.75;
nodeMeshes[2].material.color.setHex(flicker ? 0xef4444 : nodeColors[2]);
} else {
faultSegment.material.color.setHex(0x38bdf8);
faultSegment.material.opacity = 0.75;
nodeMeshes[2].material.color.setHex(nodeColors[2]);
}
renderer.render(scene, camera);
}
animId = requestAnimationFrame(animate);
} catch (e) {
showFallback();
}
})();
</script>
</body>
</html>"""
    lines = [line.strip() for line in raw_html.splitlines() if line.strip()]
    return "\n".join(lines)
