# Installation

Drone Controllers requires Python 3.11 or later.

## Install from PyPI

!!! info "Coming Soon"
    The package will be available on PyPI soon. For now, install from source.

```bash
pip install drone-controllers
```

## Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/utiasDSL/drone-controllers.git
cd drone-controllers
pip install -e .
```

## Dependencies

Drone Controllers has minimal dependencies:

- `numpy>=2.0.0` - For numerical computations
- `scipy` - For optimization and signal processing
- `array-api-compat` - For array API compatibility
- `array-api-extra` - Additional array operations
- `array-api-typing` - Type hints for array APIs

## Verify Installation

Test your installation by importing the package:

```python
import drone_controllers
print(drone_controllers.__version__)
```

## Development Installation

If you plan to contribute to Drone Controllers, install the development dependencies using [pixi](https://pixi.sh/):

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/utiasDSL/drone-controllers.git
cd drone-controllers
pixi install

# Run tests
pixi run -e tests tests
```

## What's Next?

Once you have Drone Controllers installed, check out the [Quick Start](quick-start.md) guide to learn the basics.
