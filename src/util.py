import torch
import os

from log import get_logger

logger = get_logger(__name__)


def check_gpu_mem():
    device = torch.device('cuda')
    # os.system('nvidia-smi')
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    print(f'allocated_memory: {allocated_memory}', flush=True)
    print(f'reserved_memory: {reserved_memory}', flush=True)


def slurm_job_id():
    return os.environ.get('SLURM_JOB_ID')
