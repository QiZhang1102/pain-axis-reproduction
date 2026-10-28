import unittest
from scripts.prepare_official_extraction import prepare


class TestPinnedExtractionPreparation(unittest.TestCase):
    def test_only_environment_campaign_changes(self):
        original = '''from pathlib import Path
# On RunPod, /workspace is the persistent network volume.
if Path("/workspace").exists():
    OUTPUT_DIR = Path("/workspace/results")
    print("Saving to /workspace/results")
else:
    OUTPUT_DIR = Path("results")
    print("Local environment - saving to ./results")
MODELS = [
    ("google/gemma-2-2b", "Gemma_2_2B_base"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct"),
]
def clear_gpu_and_cache():
    gc.collect()
    shutil.rmtree("/workspace/.cache")

def save_progress():
    pass
N_FOLDS=5
RANDOM_SEED=42
DENOISE_VARIANCE=0.5
'''
        prepared = prepare(original)
        self.assertNotIn("shutil.rmtree", prepared)
        self.assertIn('OUTPUT_DIR = Path("results")', prepared)
        self.assertNotIn("Gemma_2_2B_base", prepared)
        self.assertEqual(prepared.count('"Qwen/Qwen2.5-7B-Instruct"'), 1)
        for unchanged in ("N_FOLDS=5", "RANDOM_SEED=42", "DENOISE_VARIANCE=0.5"):
            self.assertIn(unchanged, prepared)
        self.assertIn("def save_progress():", prepared)

    def test_fail_closed_on_unexpected_source(self):
        with self.assertRaises(ValueError):
            prepare("MODELS = []")


if __name__ == "__main__":
    unittest.main()
