# JIT Compilation

Every controller is a pure function with no hidden state or side effects. In addition, they are implemented exclusively with Array API operations that are compatible with lazy JIT frameworks. Together, these two properties mean that every controller can be JIT compiled without any modification.

```python
import jax
import jax.numpy as jnp
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250", xp=jnp)
jit_ctrl = jax.jit(ctrl)

pos  = jnp.zeros(3)
quat = jnp.array([0., 0., 0., 1.])
vel  = jnp.zeros(3)
cmd  = jnp.zeros(13)

rpyt, int_pos_err = jit_ctrl(pos, quat, vel, cmd)
```
## Integral errors under JIT

Integral errors are regular arrays and are handled as JAX pytree leaves, so they pass through `jax.jit` without any special treatment.

```python
import jax
import jax.numpy as jnp
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl     = parametrize(state2attitude, "cf2x_L250", xp=jnp)
jit_ctrl = jax.jit(ctrl)

pos  = jnp.zeros(3)
quat = jnp.array([0., 0., 0., 1.])
vel  = jnp.zeros(3)
cmd  = jnp.zeros(13)

ctrl_errors = None
for _ in range(10):
    rpyt, int_pos_err = jit_ctrl(pos, quat, vel, cmd, ctrl_errors=ctrl_errors)
    ctrl_errors = (int_pos_err,)
```

## Batched JIT

Batching and JIT compose directly. Add leading dimensions to the state arrays and the same compiled function handles the entire batch.

```python
import jax
import jax.numpy as jnp
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl     = parametrize(state2attitude, "cf2x_L250", xp=jnp)
jit_ctrl = jax.jit(ctrl)

N    = 1_000
pos  = jnp.zeros((N, 3))
quat = jnp.broadcast_to(jnp.array([0., 0., 0., 1.]), (N, 4))
vel  = jnp.zeros((N, 3))
cmd  = jnp.zeros((N, 13))

rpyt, _ = jit_ctrl(pos, quat, vel, cmd)
rpyt.shape  # (1000, 4)
```
