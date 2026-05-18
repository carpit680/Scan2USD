.PHONY: install doctor preprocess reconstruct label lift synthesize export benchmark clean test test-e2e

install:
	python -m pip install -e ".[dev]"

doctor:
	scan2usd doctor configs/example_scene.yaml

preprocess:
	scan2usd preprocess configs/example_scene.yaml

reconstruct:
	scan2usd reconstruct configs/example_scene.yaml

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
