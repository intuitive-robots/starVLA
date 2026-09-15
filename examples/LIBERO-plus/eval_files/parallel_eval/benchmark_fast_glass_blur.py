"""Validate all severities against the pinned source inside the simulator SIF."""
import ast
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from fast_glass_blur import build_fast_glass_blur

source = Path('/app/LIBERO-plus/libero/libero/envs/env_wrapper.py')
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'glass_blur')
namespace = {'np': np, 'gaussian': gaussian}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
original = namespace['glass_blur']
fast = build_fast_glass_blur(original)
rng = np.random.RandomState(1823)
# First invocation includes compilation; exclude it from the timing table.
fast(Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)))
rows = []
for shape in [(17, 21, 3), (256, 256, 3)]:
    for severity in range(1, 11):
        for seed in [7, 43]:
            image = Image.fromarray(rng.randint(0, 256, shape, dtype=np.uint8))
            np.random.seed(seed)
            start = time.perf_counter()
            expected = original(image, severity)
            original_s = time.perf_counter() - start
            expected_state = np.random.get_state()
            np.random.seed(seed)
            start = time.perf_counter()
            actual = fast(image, severity)
            fast_s = time.perf_counter() - start
            actual_state = np.random.get_state()
            assert np.array_equal(expected, actual), (shape, severity, seed, 'pixels differ')
            assert all(np.array_equal(a, b) for a, b in zip(expected_state, actual_state)), 'RNG differs'
            rows.append(dict(shape=shape, severity=severity, seed=seed, original_s=original_s,
                             fast_s=fast_s, speedup=original_s / fast_s))
print(json.dumps({'passed': len(rows), 'comparisons': rows}, indent=2))
