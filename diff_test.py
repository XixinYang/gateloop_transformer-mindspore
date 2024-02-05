import numpy as np
from gateloop_transformer import Transformer

from mindspore import ops

model = Transformer(num_tokens=256, dim=624, depth=6, use_gate_looped_attn=True)

ids = ops.randint(0, 256, (1, 1024))
logits = model(ids)  # (1, 1024, 256)

# save input and weights
np.save("D://桌面//ids.npy", ids.asnumpy())
params = model.parameters_dict()
pt_params = {}
for name in params:
    p = params[name]
    if name.endswith(".beta"):
        name = name[: name.rfind(".beta")] + ".bias"
    if name.endswith(".gamma"):
        name = name[: name.rfind(".gamma")] + ".weight"
    if name.endswith(".moving_mean"):
        name = name[: name.rfind(".moving_mean")] + ".running_mean"
    if name.endswith(".moving_variance"):
        name = name[: name.rfind(".moving_variance")] + ".running_var"
    if name.endswith(".embedding_table"):
        name = name[: name.rfind(".embedding_table")] + ".weight"
    if "to_a" in name:
        name = name.replace("to_a", "to_a.0")
    if "norm.weight" in name:
        name = name.replace("norm.weight", "norm.gamma")
    if name == "to_logits.0.weight":
        name = name.replace("weight", "gamma")
    pt_params[name] = p.value().asnumpy()
np.save("D://桌面//params.npy", pt_params)
np.save("D://桌面//logits.npy", logits.asnumpy())

### add the following part into pytorch-version test code
# model.eval()
# params=np.load("D://桌面//params.npy",allow_pickle=True).item()
# for i in params:
#     params[i]=tensor(params[i])
# model.load_state_dict(params,strict=True)
# ids = torch.tensor(np.load("D://桌面//ids.npy"))
# logits = model(ids) # (1, 1024, 256)
# rel_diff=abs((logits.detach().numpy()-np.load("D://桌面//logits.npy"))/torch.where(logits!=0, logits, logits.mean()).detach().numpy()).mean() # 1.1224853e-05
