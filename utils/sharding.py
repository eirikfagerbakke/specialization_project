import jax
#jax.config.update("jax_enable_x64", True)

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard

def create_device_mesh():
    """Create a mesh for array sharding and autoparallelism."""
    num_devices = len(jax.devices())
    if num_devices == 1:
        print("Only one device detected. Disabling array sharding and autoparallelism.")
        return None, None, None
    else:
        print("Created mesh for array sharding and autoparallelism.")
        print(f"Using {num_devices} devices.")
        devices = mesh_utils.create_device_mesh((num_devices, 1))
        sharding_u = jshard.PositionalSharding(devices).reshape(num_devices, 1, 1)
        sharding_a = jshard.PositionalSharding(devices)
        replicated = sharding_a.replicate()

    return sharding_a, sharding_u, replicated