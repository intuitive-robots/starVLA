"""LIBERO-plus benchmark — mixtures.

Deliberately does NOT register a ``ROBOT_TYPE_CONFIG_MAP`` entry: LIBERO-plus uses
the same 7-D delta action / 8-D state layout as vanilla LIBERO, so it reuses the
``libero_franka`` DataConfig from ``examples/LIBERO/train_files/data_registry/``
(including its ``action.gripper: binary_invert`` normalization).  Registering
``libero_franka`` here too would make the winner depend on registry scan order.

The dataset lives directly at ``<data_root_dir>/libero_plus`` (data/ meta/ videos/
at the top level), so the mixture entry is the bare directory name and
``data_root_dir`` must point at its PARENT.
"""

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    # Overrides the base-registry entry, which expects a "libero_plus.0.0_lerobot"
    # sub-directory that does not exist in this layout.
    "libero_plus": [
        ("libero_plus", 1.0, "libero_franka"),
    ],
}
