"""End-to-end tests driven by the checked-in synthetic fixture."""

from __future__ import annotations

import unittest

from helpers import load_payload

from operator_embedding_shards import (
    ApplyOutcome,
    EmbeddingShards,
    ParameterStore,
    TokenStatus,
    VocabularyRegistry,
    vocabulary_from_payload,
)


class FixtureEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.payload = load_payload()
        self.vocab = vocabulary_from_payload(self.payload)
        self.registry = VocabularyRegistry([self.vocab])
        self.op = EmbeddingShards(self.registry)

    def test_forward_returns_fixture_order_vectors_status_and_shards(self):
        result = self.op.forward(self.payload["ids"])
        self.assertEqual(len(result), len(self.payload["ids"]))
        self.assertEqual(result.values[0], (1.0, 0.0))
        self.assertEqual(result.values[1], (0.0, 3.0))
        self.assertEqual(result.values[2], (0.0, 3.0))
        self.assertEqual(
            result.shard_attribution(), ("a", "b", "b", None, None)
        )
        self.assertEqual(
            [status.value for status in result.statuses()],
            ["ok", "ok", "ok", "missing", "unknown"],
        )

    def test_backward_merges_repeated_id_and_blocks_disabled(self):
        ids = self.payload["ids"] + self.payload["disabled"]
        forward = self.op.forward(ids, step=1, micro_batch_id="fixture")
        gradients = [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0), None, None, None]
        batch = self.op.backward(forward, gradients)

        by_id = {record.target.token_id: record for record in batch}
        self.assertEqual(set(by_id), {0, 5})
        self.assertEqual(by_id[5].gradient, (5.0, 1.0))
        self.assertEqual(by_id[5].occurrences, (1, 2))
        self.assertEqual(by_id[5].target.shard_id, "b")
        self.assertEqual(len(batch.blocked), 1)
        self.assertEqual(batch.blocked[0].token_id, 6)
        self.assertEqual(
            [item.token_id for item in batch.skipped],
            [9, -1],
        )

    def test_full_training_step_end_to_end(self):
        store = ParameterStore(self.registry, learning_rate=1.0)
        forward = self.op.forward(self.payload["ids"], step=1, micro_batch_id="fixture")
        gradients = [(1.0, 0.0), (0.0, 1.0), (0.0, 1.0), None, None]
        report = store.apply(self.op.backward(forward, gradients))
        self.assertEqual(len(report.applied), 2)
        # id 5 started at (0, 3) and was updated once with the summed gradient.
        self.assertEqual(store.values("2026-11-a", 5), (0.0, 5.0))
        # id 0 started at (1, 0).
        self.assertEqual(store.values("2026-11-a", 0), (2.0, 0.0))

        # The registry now serves the updated rows to the next forward pass.
        self.assertEqual(self.op.embed([5]), ((0.0, 5.0),))

    def test_quality_report_carries_disabled_reason(self):
        forward = self.op.forward([6, -1, 9])
        batch = self.op.backward(forward, [None, None, None])
        report = batch.quality_report()
        self.assertEqual(report.disabled, (6,))
        self.assertTrue(report.blocks_updates)
        self.assertTrue(
            any(reason.startswith("disabled:6:") for reason in report.reasons)
        )

    def test_disabled_row_stays_untouched_by_any_update(self):
        store = ParameterStore(self.registry)
        before = store.values("2026-11-a", 6)
        forward = self.op.forward([6], step=1, micro_batch_id="m")
        store.apply(self.op.backward(forward, [None]))
        self.assertEqual(store.values("2026-11-a", 6), before)

    def test_second_apply_of_same_batch_is_duplicate(self):
        store = ParameterStore(self.registry)
        forward = self.op.forward([0], step=1, micro_batch_id="m")
        batch = self.op.backward(forward, [(1.0, 1.0)])
        store.apply(batch)
        second = store.apply(batch)
        self.assertEqual([item.outcome for item in second], [ApplyOutcome.DUPLICATE])


if __name__ == "__main__":
    unittest.main()
