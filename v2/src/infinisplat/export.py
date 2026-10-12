"""RGB images to a native Gaussian PLY, without a demo or rasterizer."""
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from .pipeline import sha256
from .cameras import save_camera_prior
from .viewer import _validate_ply, _export_html


def export_ply(scene, output, *, html=False):
    """Export the scene; html=True additionally creates a standalone SH0 viewer."""
    if html:
        from .viewer import require_html_converter
        require_html_converter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splats = scene['splats']
    fields = {}
    for i, name in enumerate(('x', 'y', 'z')):
        fields[name] = splats['means'][:, i]
    for name in ('nx', 'ny', 'nz'):
        fields[name] = None
    for i in range(3):
        fields[f'f_dc_{i}'] = splats['sh'][:, 0, i]
        fields[f'scale_{i}'] = splats['scales'][:, i].float().log()
    fields['opacity'] = torch.logit(splats['opacities'].float().clamp(1e-6, 1-1e-6))
    for i in range(4):
        fields[f'rot_{i}'] = splats['quats'][:, i]
    count = len(splats['means'])
    data = np.empty(count, dtype=[(name, '<f4') for name in fields])
    # Preserve PyTorch transforms and exact field order. Fill cache-sized row
    # blocks so seventeen column writes do not traverse the whole PLY each time.
    columns = [(name, None if tensor is None else tensor.detach().float().cpu().numpy())
               for name, tensor in fields.items()]
    values = data.view('<f4').reshape(count, len(fields))
    for start in range(0, count, 65536):
        block = values[start:start+65536]
        for i, (name, column) in enumerate(columns):
            block[:, i] = 0 if column is None else column[start:start+65536]
        if not np.isfinite(block).all():
            for i, (name, _) in enumerate(columns):
                if not np.isfinite(block[:, i]).all():
                    raise ValueError('Nonfinite Gaussian field: ' + name)
    ply = output / 'scene.ply'
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(ply))
    validation = _validate_ply(ply, expected_count=count)
    del data, values, block, columns, fields
    np.savez(output / 'cameras.npz', c2w=scene['c2w'].cpu().numpy(),
             intrinsics=scene['intrinsics'].cpu().numpy())
    metadata = dict(scene['metadata'])
    save_camera_prior(scene, output)
    metadata['files'] = {p.name: {'sha256': validation['sha256'] if p == ply else sha256(p), 'bytes': p.stat().st_size}
                         for p in output.iterdir()}
    (output / 'result.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    if html:
        metadata['viewer'] = _export_html(ply, output / 'scene.html', validation=validation)
        metadata['files']['scene.html'] = {key: metadata['viewer'][key] for key in ('sha256', 'bytes')}
        (output / 'result.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return ply
