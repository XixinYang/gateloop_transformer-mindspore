<img src="./gateloop.png" width="450px"></img>

## GateLoop Transformer

Implementation of <a href="https://arxiv.org/abs/2311.01927">GateLoop</a> Transformer in MindSpore. Original by [lucidrains](https://github.com/lucidrains)/[gateloop-transformer](https://github.com/lucidrains/gateloop-transformer).

### Usage

```python
from mindspore import ops
from gateloop_transformer import Transformer

model = Transformer(
    num_tokens = 256,
    dim = 624,
    depth = 6,
    use_gate_looped_attn = True
)

ids = ops.randint(0, 256, (1, 1024))
logits = model(ids) # (1, 1024, 256)
```

A simplified gate loop layer

```python
from mindspore import ops
from gateloop_transformer import SimpleGateLoopLayer

gateloop = SimpleGateLoopLayer(512)

x = ops.randn((1, 65536, 512))
x = gateloop(x) + x
```
## Citations

```bibtex
@inproceedings{Katsch2023GateLoopFD,
    title   = {GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling},
    author  = {Tobias Katsch},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265018962}
}
```

```bibtex
@inproceedings{Heinsen2023EfficientPO,
    title   = {Efficient Parallelization of a Ubiquitous Sequential Computation},
    author  = {Franz A. Heinsen},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265213659}
}
```
