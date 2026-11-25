import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_embedding_fixture_distinguishes_repeat_unknown_and_disabled_ids(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        self.assertGreater(payload["ids"].count(5), 1)
        self.assertIn(payload["unknown_id"], payload["ids"])
        self.assertTrue(set(payload["disabled"]).isdisjoint(payload["ids"]))
        self.assertEqual(sum(len(shard["values"]) for shard in payload["shards"]), 8)

if __name__ == "__main__":
    unittest.main()
