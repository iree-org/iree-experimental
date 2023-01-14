import jax
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client


a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = a
for i in range(100):
  b = jax.numpy.asarray([i]) * a + b
print(b)
