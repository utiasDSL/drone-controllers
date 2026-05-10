# Installation

Select your installation method from the tabs below, then read the notes under each section for what it includes.

=== "pip"

    ```bash
    pip install drone-controllers
    ```

=== "pixi"

    ```bash
    git clone https://github.com/learnsyslab/drone-controllers.git
    cd drone-controllers
    pixi shell
    ```

=== "pixi + tests"

    ```bash
    git clone https://github.com/learnsyslab/drone-controllers.git
    cd drone-controllers
    pixi shell -e tests
    ```

---

## Developer install

[Pixi](https://pixi.sh/) creates a fully reproducible environment and installs `drone-controllers` in editable mode. Any source change takes effect immediately without reinstalling. Recommended for contributors and researchers who modify the controllers.

## Testing

Adds `pytest` and `pytest-markdown-docs` for running the test suite and doc snippet tests.

```bash
pixi run -e tests tests       # unit tests
pixi run -e tests test-docs   # doc code snippet tests
```

## Verify the installation

```python
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
list(ctrl.keywords.keys())[:3]  # ['mass', 'kp', 'kd']
```
