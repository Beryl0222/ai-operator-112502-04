"""Forward gather, backward aggregation and shard attribution tests."""

from __future__ import annotations

import unittest

from helpers import fixture_registry, fixture_vocabulary, second_version, simple_vocabulary

from operator_embedding_shards import (
    AggregationError,
    AggregationRule,
    BatchShapeError,
    EmbeddingShards,
    GradientShapeError,
    MissingTokenError,
    TokenStatus,
    VocabularyRegistry,
    aggregate_gradients,
)


def single_op(**kwargs):
    return EmbeddingShards(fixture_registry(), **kwargs)


class ForwardGatherTest(unittest.TestCase):
    def test_returns_vectors_in_original_token_order(self):
        op = single_op()
        result = op.forward([5, 0, 5])
        self.assertEqual(
            result.values,
            ((0.0, 3.0), (1.0, 0.0), (0.0, 3.0)),
        )

    def test_reports_status_and_shard_for_each_position(self):
        op = single_op()
        result = op.forward([0, 5, 9, -1])
        self.assertEqual(
            [vector.status for vector in result],
            [TokenStatus.OK, TokenStatus.OK, TokenStatus.MISSING, TokenStatus.UNKNOWN],
        )
        self.assertEqual(result.shard_attribution(), ("a", "b", None, None))

    def test_repeated_id_reuses_the_first_vector_snapshot(self):
        op = single_op()
        result = op.forward([5, 0, 5, 5])
        self.assertIsNone(result[0].reuse_of)
        self.assertEqual(result[2].reuse_of, 0)
        self.assertEqual(result[3].reuse_of, 0)
        self.assertEqual(result[0].values, result[3].values)

    def test_disabled_vector_is_emitted_but_marked_disabled(self):
        op = single_op()
        result = op.forward([6])
        self.assertIs(result[0].status, TokenStatus.DISABLED)
        self.assertEqual(result[0].shard_id, "b")

    def test_padding_position_is_distinct_from_unknown(self):
        op = EmbeddingShards(fixture_registry(), padding_id=0)
        result = op.forward([0, -1])
        self.assertIs(result[0].status, TokenStatus.PADDING)
        self.assertIs(result[1].status, TokenStatus.UNKNOWN)
        self.assertEqual(len(result.non_embeddable()), 2)

    def test_missing_ids_can_be_rejected(self):
        op = single_op(reject_missing=True)
        with self.assertRaises(MissingTokenError):
            op.forward([9])

    def test_embed_returns_plain_vectors(self):
        op = single_op()
        self.assertEqual(op.embed([0, 4]), ((1.0, 0.0), (3.0, 0.0)))

    def test_non_integer_ids_are_rejected(self):
        op = single_op()
        with self.assertRaises(BatchShapeError):
            op.forward([0, "5"])  # type: ignore[list-item]

    def test_per_position_versions_must_match_length(self):
        op = single_op()
        with self.assertRaises(BatchShapeError):
            op.forward([0, 5], versions=["2026-11-a"])

    def test_unknown_version_fails_before_emitting(self):
        op = single_op()
        with self.assertRaises(Exception):
            op.forward([0], versions="missing-version")

    def test_default_micro_batch_id_is_deterministic(self):
        op = single_op()
        first = op.forward([0, 5], step=1)
        second = op.forward([0, 5], step=1)
        self.assertEqual(first.micro_batch_id, second.micro_batch_id)
        self.assertNotEqual(first.micro_batch_id, op.forward([5, 0]).micro_batch_id)


class BackwardAggregationTest(unittest.TestCase):
    def test_repeated_ids_are_summed_by_default(self):
        op = single_op()
        forward = op.forward([5, 0, 5])
        batch = op.backward(forward, [(1.0, 2.0), (10.0, 0.0), (4.0, 5.0)])
        by_id = {record.target.token_id: record for record in batch}
        self.assertEqual(by_id[5].gradient, (5.0, 7.0))
        self.assertEqual(by_id[5].occurrences, (0, 2))
        self.assertEqual(by_id[5].aggregation, AggregationRule.SUM)

    def test_mean_rule_divides_by_occurrence_count(self):
        op = single_op(aggregation=AggregationRule.MEAN)
        forward = op.forward([5, 5])
        batch = op.backward(forward, [(1.0, 3.0), (3.0, 5.0)])
        self.assertEqual(batch.records[0].gradient, (2.0, 4.0))
        self.assertEqual(batch.records[0].occurrences, (0, 1))

    def test_first_and_last_rules_preserve_order(self):
        forward = EmbeddingShards(fixture_registry()).forward([5, 5, 5])
        first = EmbeddingShards(fixture_registry()).backward(
            forward, [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], aggregation="first"
        )
        last = EmbeddingShards(fixture_registry()).backward(
            forward, [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], aggregation="last"
        )
        self.assertEqual(first.records[0].gradient, (1.0, 0.0))
        self.assertEqual(last.records[0].gradient, (3.0, 0.0))

    def test_error_rule_rejects_repeats(self):
        op = single_op()
        forward = op.forward([5, 5])
        with self.assertRaises(AggregationError):
            op.backward(forward, [(1.0, 1.0), (1.0, 1.0)], aggregation="error")

    def test_error_rule_allows_unique_ids(self):
        op = single_op()
        forward = op.forward([0, 5])
        batch = op.backward(forward, [(1.0, 1.0), (1.0, 1.0)], aggregation="error")
        self.assertEqual(len(batch.records), 2)

    def test_unsupported_rule_is_rejected(self):
        with self.assertRaises(AggregationError):
            AggregationRule.parse("median")

    def test_gradient_dimension_must_match_embedding(self):
        op = single_op()
        forward = op.forward([0, 5])
        with self.assertRaises(GradientShapeError):
            op.backward(forward, [(1.0,), (1.0, 2.0)])

    def test_gradient_count_must_match_positions(self):
        op = single_op()
        forward = op.forward([0, 5])
        with self.assertRaises(GradientShapeError):
            op.backward(forward, [(1.0, 2.0)])

    def test_missing_gradient_for_embeddable_position_is_rejected(self):
        op = single_op()
        forward = op.forward([0, 5])
        with self.assertRaises(GradientShapeError):
            op.backward(forward, [(1.0, 2.0), None])

    def test_non_embeddable_positions_accept_no_gradient(self):
        op = single_op()
        forward = op.forward([0, 9, -1, 6])
        batch = op.backward(forward, [(1.0, 2.0), None, None, None])
        self.assertEqual([record.target.token_id for record in batch], [0])

    def test_duplicate_ids_under_different_versions_are_not_merged(self):
        registry = VocabularyRegistry([fixture_vocabulary(), second_version()])
        op = EmbeddingShards(registry, default_version="2026-11-a")
        forward = op.forward([5, 5], versions=["2026-11-a", "2026-11-b"])
        batch = op.backward(forward, [(1.0, 0.0), (2.0, 0.0)])
        self.assertEqual(len(batch.records), 2)
        self.assertEqual(
            {record.target.version for record in batch.records},
            {"2026-11-a", "2026-11-b"},
        )

    def test_aggregate_gradients_matches_records(self):
        self.assertEqual(
            aggregate_gradients([(1.0, 2.0), (3.0, 4.0)], AggregationRule.SUM, dim=2),
            (4.0, 6.0),
        )
        self.assertEqual(
            aggregate_gradients([(1.0, 2.0), (3.0, 4.0)], AggregationRule.MEAN, dim=2),
            (2.0, 3.0),
        )
        with self.assertRaises(AggregationError):
            aggregate_gradients([], AggregationRule.SUM, dim=2)


class RecordProvenanceTest(unittest.TestCase):
    def test_each_record_carries_shard_and_version_attribution(self):
        op = single_op()
        forward = op.forward([0, 5], step=7, micro_batch_id="micro-1")
        batch = op.backward(forward, [(1.0, 0.0), (0.0, 1.0)])
        record = batch.records[1]
        self.assertEqual(record.target.token_id, 5)
        self.assertEqual(record.target.shard_id, "b")
        self.assertEqual(record.target.shard_start, 4)
        self.assertEqual(record.target.local_index, 1)
        self.assertEqual(record.target.global_index, 5)
        self.assertEqual(record.target.version, "2026-11-a")
        self.assertEqual(record.micro_batch_id, "micro-1")
        self.assertEqual(record.step, 7)
        self.assertEqual(record.row_fingerprint, fixture_vocabulary().row_fingerprint(5))

    def test_update_ids_are_content_addressed(self):
        op = single_op()
        forward = op.forward([0, 5], step=1, micro_batch_id="m")
        one = op.backward(forward, [(1.0, 0.0), (0.0, 1.0)])
        same = op.backward(forward, [(1.0, 0.0), (0.0, 1.0)])
        different = op.backward(forward, [(2.0, 0.0), (0.0, 1.0)])
        self.assertEqual(one.update_ids, same.update_ids)
        self.assertNotEqual(one.update_ids, different.update_ids)

    def test_serialisation_round_trips(self):
        from operator_embedding_shards import UpdateBatch

        op = single_op()
        forward = op.forward([5, 0, 5, 6, -1], step=3, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), None, None])
        restored = UpdateBatch.from_dict(batch.as_dict())
        self.assertEqual(restored, batch)
        self.assertEqual(restored.update_ids, batch.update_ids)
        self.assertEqual(restored.blocked[0].occurrences, (3,))

    def test_tampered_serialised_update_id_is_rejected(self):
        from operator_embedding_shards import UpdateBatch

        op = single_op()
        forward = op.forward([0], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 0.0)])
        payload = batch.as_dict()
        payload["records"][0]["update_id"] = "tampered"
        with self.assertRaises(Exception):
            UpdateBatch.from_dict(payload)


class DisabledHandlingTest(unittest.TestCase):
    def test_disabled_ids_produce_blocked_updates_with_reason(self):
        op = single_op()
        forward = op.forward([6, 0, 6])
        batch = op.backward(forward, [None, (1.0, 1.0), None])
        self.assertEqual(len(batch.blocked), 1)
        blocked = batch.blocked[0]
        self.assertEqual(blocked.token_id, 6)
        self.assertEqual(blocked.occurrences, (0, 2))
        self.assertEqual(blocked.shard_id, "b")
        self.assertIn("disabled", blocked.reason)
        self.assertEqual([record.target.token_id for record in batch], [0])

    def test_quality_report_distinguishes_all_statuses(self):
        vocab = simple_vocabulary(
            disabled=frozenset({4}), unknown_id=-1
        )
        registry = VocabularyRegistry([vocab])
        op = EmbeddingShards(registry, padding_id=0)
        forward = op.forward([0, -1, 4, 9])
        batch = op.backward(forward, [None, None, None, None])
        report = batch.quality_report()
        self.assertEqual(report.padding, (0,))
        self.assertEqual(report.unknown, (-1,))
        self.assertEqual(report.disabled, (4,))
        self.assertEqual(report.missing, (9,))
        self.assertTrue(report.blocks_updates)
        self.assertTrue(report.has_issues)
        self.assertEqual(report.counts["disabled"], 1)

    def test_blocked_updates_are_not_updatable_records(self):
        op = single_op()
        forward = op.forward([6])
        batch = op.backward(forward, [None])
        self.assertEqual(len(batch), 0)
        self.assertEqual(batch.update_ids, ())


if __name__ == "__main__":
    unittest.main()
