"""Export a validated Gaussian PLY as a standalone SH0 HTML viewer."""
import json
from functools import lru_cache
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from plyfile import PlyData
from .pipeline import sha256

def _finite_fields(vertices, fields, chunk_size=65536):
    """Scan bounded contiguous chunks rather than rereading every strided column."""
    names = vertices.dtype.names
    packed = names and vertices.dtype.itemsize == len(names) * 4 and all(
        vertices.dtype.fields[name][0] == np.dtype('<f4')
        and vertices.dtype.fields[name][1] == i * 4 for i, name in enumerate(names))
    if packed:
        values = vertices.view('<f4').reshape(len(vertices), len(names))
        for start in range(0, len(vertices), chunk_size):
            block = values[start:start+chunk_size]
            if not np.isfinite(block).all():
                # Unused external PLY properties need not be finite.
                if any(not np.isfinite(vertices[name][start:start+chunk_size]).all() for name in fields):
                    return False
        return True
    return all(np.isfinite(vertices[name]).all() for name in fields)


def _file_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _validate_ply(ply, expected_count=None):
    """Independent reader and hash, bound to this exact file generation."""
    ply = Path(ply).resolve()
    before = _file_signature(ply)
    vertices = PlyData.read(str(ply), mmap=True)['vertex'].data
    required = ['x','y','z','opacity'] + [f'{prefix}_{i}' for prefix,n in
                 [('scale',3),('rot',4),('f_dc',3)] for i in range(n)]
    if not len(vertices) or not set(required).issubset(vertices.dtype.names):
        raise ValueError('Expected a nonempty Gaussian PLY with scale, rotation, opacity and SH0')
    if expected_count is not None and len(vertices) != expected_count:
        raise ValueError('PLY validation failed: Gaussian count differs')
    fields = required + [k for k in vertices.dtype.names if k.startswith('f_rest_')]
    if not _finite_fields(vertices, fields):
        raise ValueError('Nonfinite Gaussian PLY payload')
    count = len(vertices)
    del vertices
    digest = sha256(ply)
    if _file_signature(ply) != before:
        raise RuntimeError('PLY changed during validation')
    return dict(path=ply, signature=before, gaussian_count=count, sha256=digest)


def _export_workers():
    value = os.environ.get('INFINISPLAT_EXPORT_WORKERS')
    if value is not None:
        workers = int(value)
        if not 0 <= workers <= 8:
            raise ValueError('INFINISPLAT_EXPORT_WORKERS must be from 0 to 8')
        return workers
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            available = min(available, max(1, int(quota)//int(period)))
    except (OSError, ValueError):
        pass
    return max(1, min(8, available))


@lru_cache(maxsize=8)
def _cached_version(executable, signature):
    result = subprocess.run([executable, '--version'], check=True, capture_output=True, text=True)
    version = (result.stdout or result.stderr).strip()
    if not version:
        raise RuntimeError('Cannot determine splat-transform version')
    return version


def _converter_version(executable):
    path = Path(executable).resolve()
    if not path.is_file():
        # Test doubles/non-files cannot provide a stable cache identity.
        return _cached_version.__wrapped__(executable, None)
    signature = [_file_signature(path)]
    module = path.parent.parent/'dist/cli.mjs'
    if module.is_file():
        signature.append(_file_signature(module))
    return _cached_version(executable, tuple(signature))


def _viewer_settings():
    return dict(version=2, tonemapping='none', highPrecisionRendering=False,
                background=dict(color=[0.4, 0.4, 0.4]),
                postEffectSettings=dict(
                    sharpness=dict(enabled=False, amount=0),
                    bloom=dict(enabled=False, intensity=1, blurLevel=2),
                    grading=dict(enabled=False, brightness=0, contrast=1,
                                 saturation=1, tint=[1, 1, 1]),
                    vignette=dict(enabled=False, intensity=0.5, inner=0.3,
                                  outer=0.75, curvature=1),
                    fringing=dict(enabled=False, intensity=0.5)),
                animTracks=[], cameras=[dict(initial=dict(
                    position=[2, 2, -2], target=[0, 0, 0], fov=75))],
                annotations=[], startMode='default')


def require_html_converter():
    executable = shutil.which('splat-transform')
    if executable is None:
        raise RuntimeError('HTML export requires splat-transform. Install it with '
                           '`npm install -g @playcanvas/splat-transform@2.6.0` '
                           'or use --no-html for PLY only.')
    return executable


def patch_viewer_html(output, display_transform=None):
    """Disable screen-space small-splat culling without changing embedded SOG data.

    An optional presentation transform replaces the built-in Z rotation. Its
    quaternion is [x,y,z,w]; translation is in viewer coordinates. Native PLY
    coordinates are never changed. Fail closed when the pinned template changes.
    """
    output = Path(output)
    data = output.read_bytes()
    marker = b'// InfiniSplat full-detail viewer v1'
    anchor = b'const { app, camera, renderer } = await createApp(canvas, config);'
    if marker not in data:
        if data.count(anchor) != 1:
            raise RuntimeError('Unsupported standalone viewer template; use splat-transform 2.6.0')
        patch = (anchor + b'\n    ' + marker +
                 b'\n    app.scene.gsplat.minPixelSize = 0;'
                 b'\n    app.scene.gsplat.minContribution = 0;')
        data = data.replace(anchor, patch, 1)
    # The viewer reapplies this setting after loading and on quality changes.
    # Patching only createApp is insufficient: that callback would restore it.
    old = b'gsplat.minContribution = 1;'
    new = b'gsplat.minContribution = 0;'
    if data.count(old) == 1:
        data = data.replace(old, new, 1)
    elif data.count(new) != 2:
        raise RuntimeError('Unsupported viewer quality callback template')
    if display_transform is not None:
        q = np.asarray(display_transform['quaternion'], dtype=float)
        t = np.asarray(display_transform['translation'], dtype=float)
        if (q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all()
                or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-6)):
            raise ValueError('Expected a finite unit xyzw quaternion and xyz translation')
        old = b'entity.setLocalEulerAngles(0, 0, 180);'
        if data.count(old) != 1:
            raise RuntimeError('Unsupported viewer display transform template')
        replacement = ('entity.setLocalRotation(' + ', '.join(map(str, q)) + ');\n'
                       '            entity.setLocalPosition(' + ', '.join(map(str, t)) + ');')
        data = data.replace(old, replacement.encode(), 1)
        if 'camera' in display_transform:
            old = b'const animTrack = getAnimTrack(resetCamera, isObjectExperience);'
            if data.count(old) != 1:
                raise RuntimeError('Unsupported viewer camera animation template')
            data = data.replace(old, b"const animTrack = settings.startMode === 'static' ? null : getAnimTrack(resetCamera, isObjectExperience);", 1)
    output.write_bytes(data)


def export_html(ply, output=None, *, display_transform=None):
    """Convert an existing PLY without filtering, subsampling or changing its space."""
    return _export_html(ply, output, display_transform=display_transform)


def _export_html(ply, output=None, validation=None, display_transform=None):
    """Internal producer path may reuse its independent reader attestation."""
    ply = Path(ply).resolve()
    output = Path(output).absolute() if output is not None else ply.with_suffix('.html')
    if output.suffix.lower() != '.html':
        raise ValueError('HTML output must end in .html')
    if output.exists():
        raise FileExistsError(output)
    executable = require_html_converter()
    if validation is None:
        validation = _validate_ply(ply)
    # Some filesystems retain identical stat timestamps for rapid same-size writes.
    # Bind reuse to content as well as metadata before invoking the converter.
    if (validation['path'] != ply or validation['signature'] != _file_signature(ply)
            or validation['sha256'] != sha256(ply)):
        raise RuntimeError('PLY changed after validation')
    count = validation['gaussian_count']
    source_hash = validation['sha256']
    version = _converter_version(executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = _viewer_settings()
    if display_transform is not None and 'camera' in display_transform:
        camera = display_transform['camera']
        position = np.asarray(camera['position'], dtype=float)
        target = np.asarray(camera['target'], dtype=float)
        fov = float(camera['fov'])
        if (position.shape != (3,) or target.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(target).all()
                or np.linalg.norm(position - target) < 1e-8 or not 0 < fov < 180):
            raise ValueError('Invalid display-frame camera')
        settings['cameras'] = [dict(initial=dict(position=position.tolist(), target=target.tolist(), fov=fov))]
        settings['startMode'] = 'static'
    try:
        with tempfile.TemporaryDirectory(prefix='infinisplat-viewer-') as temporary:
            settings_path = Path(temporary) / 'settings.json'
            settings_path.write_text(json.dumps(settings))
            command = [executable, '-w', str(ply), '-H', '0',
                       '-E', str(settings_path), '--max-workers', str(_export_workers()), str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
        if (_file_signature(ply) != validation['signature']
                or sha256(ply) != validation['sha256']):
            raise RuntimeError('PLY changed during HTML conversion')
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError('Converter did not produce a nonempty HTML')
        patch_viewer_html(output, display_transform)
        with output.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if (not any(data.find(marker) >= 0 for marker in (b'SuperSplat',b'PlayCanvas',b'sog'))
                    or not any(data.find(marker) >= 0 for marker in (b'WebGL',b'webgl'))
                    or data.find(b'cameras') < 0):
                raise RuntimeError('Standalone viewer marker check failed')
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as error:
        output.unlink(missing_ok=True)
        detail = (error.stderr or error.stdout or str(error)) if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise RuntimeError(f'HTML conversion failed; PLY retained at {ply}. {detail[-2000:]}') from error
    return dict(file=output.name, sh_degree=0, standalone=True, gaussian_count=count,
                source_ply_sha256=source_hash, sha256=sha256(output),
                bytes=output.stat().st_size, converter_version=version,
                viewer_settings=settings, small_splat_culling=False,
                display_transform=display_transform)
