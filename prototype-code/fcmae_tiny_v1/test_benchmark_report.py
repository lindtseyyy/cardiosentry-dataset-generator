"""Check checkpoint-aware estimates and portable comparison reports on CPU."""
import json
import tempfile
import unittest
from pathlib import Path
from benchmark_report import project, projections, write_reports, comparison_rows


class BenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(batch_size=2, accumulation=2, epochs=20, save_every=10,
                        checkpoint_every_updates=2, checkpoint_every_minutes=0)

    def test_short_group_and_global_update_checkpoint_count(self):
        # Ten images: updates of 4, 4, 2. Epoch one saves at 2/3;
        # epoch two saves at global steps 4/6, plus one completion each.
        r = project(self.cfg, 10, 2, 1, 2, 3, 5, 7, 2, 4)
        self.assertEqual(r['checkpoint_writes'], 6)
        self.assertEqual(r['encoder_exports'], 3)
        self.assertEqual(r['training_seconds'], 20)
        self.assertEqual(r['validation_seconds'], 8)
        self.assertAlmostEqual(r['steady_total_hours'], (20+8+18+15)/3600)
        self.assertAlmostEqual(r['steady_compute_units'], 61/3600*4)

    def test_timer_and_final_epoch_artifacts(self):
        cfg = dict(self.cfg, epochs=1, checkpoint_every_updates=0,
                   checkpoint_every_minutes=5/60)
        # Timer triggers after eight seconds; final short group still saves.
        r = project(cfg, 10, 2, 1, 1, 3, 5, 7, 1)
        self.assertEqual(r['checkpoint_writes'], 4)
        self.assertEqual(r['encoder_exports'], 3)
        self.assertEqual(r['preview_writes'], 1)
        self.assertIsNone(r['steady_compute_units'])

    def test_zero_rate_is_distinct_from_unknown(self):
        args = (self.cfg, 10, 2, 1, 1, 1, 1, 1, 1)
        self.assertIsNone(project(*args)['steady_compute_units'])
        self.assertEqual(project(*args, units_per_hour=0)['steady_compute_units'], 0)
        for rate in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                project(*args, units_per_hour=rate)

    def test_preparation_charged_once_and_observed_range(self):
        rows = projections(self.cfg, {'ssl_train':10, 'ssl_validation':2},
                           [1,2,3], [1,2,3],
                           dict(checkpoint_seconds=0, export_seconds=0, preview_seconds=0),
                           600, 1200, 2)
        self.assertEqual(len(rows), 6)  # Horizons 1 and 20, no duplicate 20.
        for r in rows:
            self.assertAlmostEqual(r['planning_total_hours']-r['steady_total_hours'], .5)
            self.assertAlmostEqual(r['planning_compute_units'], r['planning_total_hours']*2)
        means = [r for r in rows if r['scenario']=='mean_observed']
        self.assertAlmostEqual(means[1]['training_seconds'], 400)
        self.assertLess(rows[0]['steady_total_hours'], rows[1]['steady_total_hours'])
        self.assertLess(rows[1]['steady_total_hours'], rows[2]['steady_total_hours'])

    def test_reports_preserve_comparison_groups_and_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = projections(self.cfg, {'ssl_train':10, 'ssl_validation':2}, [1], [1],
                               dict(checkpoint_seconds=0, export_seconds=0, preview_seconds=0),
                               0, 0, None)
            write_reports(dict(status='complete', gpu_name='T4', comparison_key='one',
                               units_per_hour=None, projections=rows), root/'t4')
            write_reports(dict(status='failed', gpu_name='L4', comparison_key='two',
                               error='CUDA out of memory'), root/'l4')
            result = comparison_rows(root)
            self.assertEqual({r['comparison_key'] for r in result}, {'one','two'})
            failed = next(r for r in result if r['status']=='failed')
            self.assertIsNone(failed['planned_hours'])
            self.assertIn('not supplied', (root/'t4/summary.md').read_text())
            self.assertTrue((root/'t4/estimates.csv').exists())
            self.assertEqual(json.loads((root/'l4/report.json').read_text())['error'], 'CUDA out of memory')


if __name__ == '__main__':
    unittest.main()
