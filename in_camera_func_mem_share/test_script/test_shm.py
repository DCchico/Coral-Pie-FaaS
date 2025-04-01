import time
import numpy as np
from multiprocessing import shared_memory
import os

# Get instance ID from environment (to distinguish containers)
instance_id = os.environ.get("INSTANCE_ID", "0")

if instance_id == "1":
    # First container creates shared memory
    print(f"Container {instance_id}: Creating shared memory")
    
    # Create data
    data = np.array([1, 2, 3, 4, 5])
    
    # Create shared memory
    shm = shared_memory.SharedMemory(name="test_shm", create=True, size=data.nbytes)
    
    # Create array in shared memory
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    
    print(f"Container {instance_id}: Created shared array with values: {shared_array}")
    print(f"Container {instance_id}: Waiting...")
    
    # Wait for other container to read
    time.sleep(60)
    
    # Clean up
    shm.close()
    shm.unlink()
    
else:
    # Second container reads shared memory
    print(f"Container {instance_id}: Waiting for shared memory...")
    
    # Wait for first container to create
    time.sleep(5)
    
    try:
        # Connect to shared memory
        shm = shared_memory.SharedMemory(name="test_shm")
        
        # Create array view
        shared_array = np.ndarray((5,), dtype=np.int64, buffer=shm.buf)
        
        print(f"Container {instance_id}: Read shared array with values: {shared_array}")
        
        # Modify the array
        shared_array[0] = 100
        print(f"Container {instance_id}: Modified array: {shared_array}")
        
        # Clean up
        shm.close()
    except Exception as e:
        print(f"Container {instance_id}: Error: {e}") 