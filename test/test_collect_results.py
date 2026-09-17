"""Regression checks for protocol/seed mistakes that change paper conclusions."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('collector', Path(__file__).parents[1] / 'scripts/collect_results.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    def test_both_libero_shapes_and_failed_shards(self):
        root = self.root / 'playground/Checkpoints/ervla_w_encoder/results'
        suites = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
        counts = dict(total_count=1000, success_count=750, success_rate=.75)
        self.write(root/'libero/overall_results.json', {s: counts for s in suites})
        self.write(root/'libero-plus/overall_results.json', {s: {'overall': counts} for s in suites})
        self.write(root/'libero-plus/libero_10/overall_results.json', {'overall': counts, 'Camera': counts})
        rows = c.collect_libero(self.root)
        self.assertEqual(len(rows), 9)  # category must not become a pseudo-suite
        self.assertEqual(sum(r['is_full_protocol']=='True' for r in rows), 8)
        (root/'libero-plus/failed_shards_libero_10.txt').write_text('0 1000 0 4\n')
        rows = c.collect_libero(self.root)
        self.assertTrue(all(r['is_full_protocol']=='False' for r in rows if r['benchmark']=='libero-plus'))
    def test_robocasa_seed_task_and_protocol(self):
        run = self.root/'playground/Checkpoints/ervla_robocasa365_pi_causal_s4'
        base = run/'checkpoints/steps_50000_pytorch_model.eval'
        data = {'env':'robocasa/OpenDrawer','seed':42,'successes':[True,False], 'success_rate':.5}
        self.write(base/'robocasa_OpenDrawer_scale_n16_g0.json',data)
        self.write(base/'robocasa_OpenDrawer_scale_n24_g0.json',data)
        rows = c.collect_robocasa(self.root)
        self.assertEqual({r['seed'] for r in rows},{'4'})
        self.assertEqual({r['suite_or_task'] for r in rows},{'OpenDrawer'})
        self.assertEqual(len({r['result_dir'] for r in rows}),2)
        self.assertEqual({r['is_full_protocol'] for r in rows},{'False'})
    def test_invalid_outcome_is_not_truthy_success(self):
        path=self.root/'playground/Checkpoints/run/checkpoints/steps_50000_pytorch_model.eval/robocasa_OpenDrawer.json'
        self.write(path,{'env':'robocasa/OpenDrawer','successes':['false']})
        with self.assertRaises(ValueError):c.collect_robocasa(self.root)
    def test_training_config_and_control_arm(self):
        path=self.root/'playground/Checkpoints/ervla_w/config.full.yaml'
        path.parent.mkdir(parents=True);path.write_text('seed: 43\n')
        self.assertEqual(c.training_seed(self.root,'ervla_w'),'43')
        self.assertEqual(c.arm_of('ervla_zbase_pi_sharedz_control'),'encoder/shared-z-off-control')
        self.assertNotEqual(c.arm_of('ervla_k2_pi_cam3d_cot05_pifix_nolatent'),'causal')
if __name__=='__main__':unittest.main()
