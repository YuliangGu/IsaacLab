import os, sys
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)


import torch
from myrl.byol_utils import BYOLSeq
from myrl.ppo_byol_shared import ObsEncoder, ActionEncoder

B,W,D,A = 8, 32, 60, 8
enc = ObsEncoder(D, 64)
aenc = ActionEncoder(A, 64)
byol = BYOLSeq(enc, z_dim=128, proj_dim=128, tau=0.99, action_encoder=aenc)


with torch.inference_mode(True):
    v1 = (torch.randn(B,W,D), torch.randn(B,W,A))
    v2 = (torch.randn(B,W,D), torch.randn(B,W,A))

# Online pass must succeed with grads
opt = torch.optim.Adam(list(byol.parameters()), lr=1e-3)
with torch.inference_mode(False):
    with torch.enable_grad():
        loss = byol.loss(v1, v2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
print("OK", float(loss))
