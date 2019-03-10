from io import BytesIO
import subprocess
import pandas as pd

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                        names=['memory.used', 'memory.free'],
                        skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('Using GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx