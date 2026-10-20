"""
check_gpu.py

Run this before train_bias_model.py. If CUDA isn't detected here, training
will silently fall back to CPU - much slower, with no error to warn you.
Fix the torch install first if this prints False.
"""

import torch
print("CUDA available" , torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :",  torch.cuda.get_device_name(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.2f} GB")
else:
     print("No GPU detected - reinstall torch with the CUDA index URL (see comment above).")