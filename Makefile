.PHONY: install doctor preprocess reconstruct init-usd segment-usd review-usd build-usd validate-usd label lift synthesize export benchmark clean test test-e2e gui gui-install gui-backend gui-frontend gui-lan

install:
	python -m pip install -e ".[dev,geometry,review]"

gui-install:
	python -m pip install -e gui/backend
	cd gui/frontend && npm install

gui-backend:
	cd gui/backend && python -m scan2usd_gui --reload

gui-frontend:
	cd gui/frontend && npm run dev

gui:
	@echo "Starting API :8765 and Vite :5173 (Ctrl+C stops both)"
	@(cd gui/backend && python -m scan2usd_gui --reload) & \
	(cd gui/frontend && npm run dev); \
	wait

gui-lan:
	@echo "LAN mode (no --reload): API 0.0.0.0:8765 + Vite :5173 — safer for long reconstruct jobs"
	@(cd gui/backend && python -m scan2usd_gui --host 0.0.0.0 --port 8765) & \
	(cd gui/frontend && npm run dev); \
	wait

doctor:
	scan2usd doctor configs/example_scene.yaml

preprocess:
	scan2usd preprocess configs/example_scene.yaml

reconstruct:
	scan2usd reconstruct configs/example_scene.yaml

init-usd:
	scan2usd init-usd configs/example_scene.yaml --mode production

segment-usd:
	scan2usd segment-usd configs/example_scene.yaml

review-usd:
	scan2usd review configs/example_scene.yaml

build-usd:
	scan2usd build-usd configs/example_scene.yaml

validate-usd:
	scan2usd validate-usd configs/example_scene.yaml

label:
	scan2usd label configs/example_scene.yaml

lift:
	scan2usd lift configs/example_scene.yaml

synthesize:
	scan2usd synthesize configs/example_scene.yaml

export:
	scan2usd export-dataset configs/example_scene.yaml --mode mixed

benchmark:
	scan2usd benchmark configs/example_scene.yaml --experiment all

clean-light:
	scan2usd clean configs/example_scene.yaml --tier light -y

clean-medium:
	scan2usd clean configs/example_scene.yaml --tier medium -y

clean-full:
	scan2usd clean configs/example_scene.yaml --tier full -y

test:
	python -m pytest -q -m "not e2e"

test-e2e:
	python -m pytest -q -m e2e
