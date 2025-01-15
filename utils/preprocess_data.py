import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_dataloader as jdl
import numpy as np

class MinMaxStandardizer:
    def __init__(self, x, min=None, max=None):
        self.x_min = min or jnp.min(x)
        self.x_max = max or jnp.max(x)
        
    def encode(self, x):
        x_s = (x - self.x_min) / (self.x_max - self.x_min)
        return x_s
    
    def decode(self, x_s):
        x = self.x_min + x_s * (self.x_max - self.x_min)
        return x

class GaussianStandardizer:
    def __init__(self, x, eps=0.00001):
        self.mean = jnp.mean(x)
        self.std = jnp.std(x) + eps

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean
    
def train_val_test_split(data, train_ratio, val_ratio, test_ratio):
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be equal to 1"
    
    NUM_SAMPLES = data.shape[0]

    train_split_idx = int(NUM_SAMPLES*train_ratio)
    val_split_idx = int(NUM_SAMPLES*(train_ratio+val_ratio))

    train, val, test = jnp.split(data, [train_split_idx, val_split_idx])
    
    #a_train = train[:, 0]
    #u_train = train

    #a_val = val[:,0]
    #u_val = val

    #a_test = test[:,0]
    #u_test = test
    
    #train_val_test = {
    #    "a_train": a_train,
    #    "u_train": u_train,
    #    "a_val": a_val,
    #    "u_val": u_val,
    #    "a_test": a_test,
    #    "u_test": u_test
    #}
    train_val_test = {
        "train" : train,
        "val" : val,
        "test" : test
    }
    
    return train_val_test

def scale_data(x, t, train_val_test):
    # scale x and t to [0, 1]
    x_standardizer = GaussianStandardizer(x)
    t_standardizer = GaussianStandardizer(t)
    x_test_s = x_standardizer.encode(x)
    t_test_s = t_standardizer.encode(t)
    
    # the trainig grid is downsampled from 256 x 256*3/2 to 64 x 64
    x_train_s = x_test_s[::4]
    t_train_s = t_test_s[:len(t)*2//3:4]
    
    u_standardizer = GaussianStandardizer(train_val_test["train"])

    u_train_s = u_standardizer.encode(train_val_test["train"])[:,:len(t)*2//3:4, ::4]
    a_train_s = u_train_s[:, 0]
    
    u_val_s = u_standardizer.encode(train_val_test["val"])[:,:len(t)*2//3:4, ::4]
    a_val_s = u_val_s[:, 0]
    
    u_test_s = u_standardizer.encode(train_val_test["test"])
    a_test_s = u_test_s[:, 0]
    
    scaled_data = {
        "x_train_s": x_train_s,
        "t_train_s": t_train_s,
        "x_test_s": x_test_s,
        "t_test_s": t_test_s,
        "a_train_s": a_train_s,
        "u_train_s": u_train_s,
        "a_val_s": a_val_s,
        "u_val_s": u_val_s,
        "a_test_s": a_test_s,
        "u_test_s": u_test_s,
        "x_mean" : x_standardizer.mean,
        "t_mean" : t_standardizer.mean,
        "u_mean" : u_standardizer.mean,
        "x_std" : x_standardizer.std,
        "t_std" : t_standardizer.std,
        "u_std" : u_standardizer.std,
    }
    
    return scaled_data
    
def add_noise(data, noise_std):
    return data + noise_std * jax.random.normal(jax.random.key(0), data.shape)

class Dataset(jdl.Dataset):
    """A custom dataset class, which loads a, u, x and t."""

    def __init__(
        self,
        x,
        t,
        *arrays: jax.Array,
        asnumpy: bool = False, # Store arrays as numpy arrays if True; otherwise store as array type of *arrays
    ):
        assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays), \
            "All arrays must have the same dimension."
        self.arrays = tuple(arrays)
        if asnumpy:
            self.asnumpy()
        self.x = x 
        self.t = t            
    
    def asnumpy(self):
        """Convert all arrays to numpy arrays."""
        self.arrays = tuple(np.array(arr) for arr in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        return jax.tree_util.tree_map(lambda x: x[index], self.arrays), self.x, self.t