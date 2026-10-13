# Component provenance

InfiniSplat v2 is the project release; `infinisplat` is the package name. This package has no Mirror backend,
Mirror backbone, Mirror camera/depth prediction heads, prior-token encoder or
Mirror transformer stack.

The following small components remain necessary to load the current Gaussian
checkpoint and preserve its numerical behavior:

- `decoder.py`: the inference subset of the existing Gaussian feature decoder,
  DPT-style fusion blocks and positional-grid utilities. The unused dense prediction
  head and activation-checkpointing options were removed.
- `geometry.py`: context-camera normalization, alignment, point-radius normalization
  and depth unprojection. Unused camera-alignment and output-access helpers were removed.
- `swiglu.py`: the PyTorch SwiGLU implementation used by the DA3 backbone.

These subsets were adapted from the existing WorldGS / WorldMirror-derived utility
files. Their original component notices are retained in `Notice.txt` and
`License.txt`; deleting the alternate backend does not erase this source provenance.
The original files also reference DINO/DINOv2, VGGT and DPT components. The retained
notices are component provenance, not a description of InfiniSplat's model architecture.

Depth Anything 3 is an external, separately installed dependency. `da3_order.py`
is an Apache-2.0 adaptation preserving input order in auxiliary feature outputs;
see `LICENSE_DA3`. The weight bundle and upstream backbone have their own terms.
PlayCanvas splat-transform is separately installed for optional HTML viewing.

Public redistribution of this local candidate and its weights has not been marked
approved. No weights are included in this code export. Selected public dataset samples
are included under `examples/data/`; see `examples/NOTICE.md` for their sources
and separate data terms.
