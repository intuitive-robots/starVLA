"""Tests for the compact SPARC DROID CoT path."""

import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from starVLA.dataloader.cot_augmentation import _rewrite_coordinate_text
from starVLA.dataloader.cot_resolver import SparcSQLiteCoTResolver


def _blob(value, dtype):
    return zlib.compress(np.asarray(value, dtype=dtype).tobytes())


class SparcSQLiteCoTResolverTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "mapping.sqlite"
        db = sqlite3.connect(self.path)
        db.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE annotations(
              annotation_id TEXT PRIMARY KEY, source_uuid TEXT, trajectory_id TEXT,
              subtask_index INTEGER, camera_view TEXT, start_frame INTEGER,
              end_frame INTEGER, image_h INTEGER, image_w INTEGER, instruction TEXT,
              object_label TEXT, target_label TEXT, initial_box_json TEXT,
              target_box_json TEXT, anchor_boxes_json TEXT, trace_frames_blob BLOB,
              trace_xy_blob BLOB, selection_score REAL, target_verified INTEGER);
            CREATE TABLE rewrite_groups(
              rewrite_key TEXT PRIMARY KEY, source_uuid TEXT, subtask_index INTEGER,
              phase_index INTEGER, instruction TEXT, phase_type TEXT, description TEXT,
              object_label TEXT, target_label TEXT, variants_json TEXT, status TEXT,
              attempts INTEGER, error TEXT);
            CREATE TABLE phases(annotation_id TEXT,phase_index INTEGER,phase_type TEXT,
              start_frame INTEGER,end_frame INTEGER,description TEXT,rewrite_key TEXT);
            CREATE TABLE annotation_rewrites(annotation_id TEXT PRIMARY KEY,
              variants_by_phase_json TEXT,status TEXT,attempts INTEGER,error TEXT);
            CREATE TABLE episode_motion(source_uuid TEXT PRIMARY KEY,num_frames INTEGER,
              xyz_zlib BLOB,gripper_zlib BLOB);
        """)
        db.executemany("INSERT INTO metadata VALUES(?,?)", [
            ("schema_version", "sparc-cot-sqlite-v1"),
            ("human_prompt", "Do {instruction}."),
            ("action_horizon", "20"),
            ("waypoints", "5"),
        ])
        db.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "ann", "episode", "0/0", 0, "observation.images.right_external",
            0, 30, 100, 200, "put block in tray", "block", "tray", None,
            json.dumps([100, 40, 160, 80]),
            json.dumps([[0, [10, 10, 30, 30]], [30, [20, 20, 40, 40]]]),
            _blob([0, 10, 20, 30], "<i4"),
            _blob([[20, 20], [40, 30], [60, 40], [80, 50]], "<f4"), 0.99, 1,
        ))
        db.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "ann-left", "episode", "0/0", 0, "observation.images.left_external",
            0, 30, 100, 200, "put block in tray", "block", "tray", None,
            json.dumps([20, 10, 60, 30]),
            json.dumps([[0, [100, 10, 140, 30]], [30, [120, 20, 160, 40]]]),
            _blob([0, 10, 20, 30], "<i4"),
            _blob([[40, 20], [50, 25], [60, 30], [70, 35]], "<f4"), 0.99, 1,
        ))
        variants = [{"subtask": "move the block to the tray",
                     "reasoning": "The block must follow the grounded path into the tray."}]
        db.execute("INSERT INTO rewrite_groups VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "rewrite", "episode", 0, 0, "put block in tray", "interact",
            "move block", "block", "tray", json.dumps(variants), "success", 1, None,
        ))
        db.execute("INSERT INTO phases VALUES(?,?,?,?,?,?,?)",
                   ("ann", 0, "interact", 0, 30, "move block", "rewrite"))
        db.execute("INSERT INTO phases VALUES(?,?,?,?,?,?,?)",
                   ("ann-left", 0, "interact", 0, 30, "move block", "rewrite"))
        phase_variants = json.dumps([{"phase_index": 0, "variants": variants}])
        db.execute("INSERT INTO annotation_rewrites VALUES(?,?,?,?,?)",
                   ("ann", phase_variants, "success", 1, None))
        db.execute("INSERT INTO annotation_rewrites VALUES(?,?,?,?,?)",
                   ("ann-left", phase_variants, "success", 1, None))
        xyz = np.zeros((31, 3), dtype=np.float32)
        xyz[:, 0] = np.linspace(0, 0.10, 31)
        db.execute("INSERT INTO episode_motion VALUES(?,?,?,?)",
                   ("episode", 31, _blob(xyz, "<f4"), _blob(np.zeros(31), "<f4")))
        db.commit()
        db.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_renders_phase_rewrite_geometry_and_motion(self):
        resolver = SparcSQLiteCoTResolver(
            str(self.path), camera_view="observation.images.right_external",
            variant_mode="first", require_rewrite=True)
        conversation = resolver.resolve("episode", 10)
        text = conversation[1]["value"]
        self.assertEqual(conversation[0]["value"], "Do {instruction}.")
        self.assertIn("<subtask>move the block to the tray</subtask>", text)
        self.assertIn('<trajectory type="object">', text)
        self.assertIn("<movement>move forward", text)
        self.assertIn("<reasoning>The block must follow", text)

    def test_trajectory_attribute_survives_coordinate_augmentation(self):
        text = '<trajectory type="object">(100,200) (300,400)</trajectory>'
        rewritten = _rewrite_coordinate_text(
            text, left=5, top=5, crop_width=90, crop_height=90,
            image_width=100, image_height=100)
        self.assertIn('<trajectory type="object">', rewritten)
        self.assertNotEqual(text, rewritten)

    def test_dynamic_camera_uses_matching_geometry(self):
        resolver = SparcSQLiteCoTResolver(str(self.path), variant_mode="first")
        right = resolver.resolve(
            "episode", 10, camera_view="observation.images.right_external"
        )[1]["value"]
        left = resolver.resolve(
            "episode", 10, camera_view="observation.images.left_external"
        )[1]["value"]
        self.assertNotEqual(right, left)
        self.assertIn("<cam1>", right)
        self.assertIn("<cam1>", left)

    def test_multicamera_mapping_requires_selected_camera(self):
        resolver = SparcSQLiteCoTResolver(str(self.path))
        with self.assertRaisesRegex(ValueError, "requires the camera_view selected"):
            resolver.resolve("episode", 10)

    def test_failed_rewrite_is_never_used(self):
        db = sqlite3.connect(self.path)
        db.execute("UPDATE annotation_rewrites SET status='failed' WHERE annotation_id='ann'")
        db.commit()
        db.close()

        resolver = SparcSQLiteCoTResolver(
            str(self.path), camera_view="observation.images.right_external",
            require_rewrite=True,
        )
        self.assertIsNone(resolver.resolve("episode", 10))

    def test_selection_score_gate_is_strict(self):
        db = sqlite3.connect(self.path)
        db.execute("UPDATE annotations SET selection_score=0.94 WHERE annotation_id='ann'")
        db.commit()
        db.close()

        resolver = SparcSQLiteCoTResolver(
            str(self.path), camera_view="observation.images.right_external",
            min_selection_score=0.94,
        )
        self.assertIsNone(resolver.resolve("episode", 10))
        self.assertEqual(resolver.num_trajectories, 1)  # left-camera success remains


if __name__ == "__main__":
    unittest.main()
