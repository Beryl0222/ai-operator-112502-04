"""Optimiser hand-off tests: applying, attribution loss, checkpoints, resume."""

from __future__ import annotations

import unittest

from helpers import fixture_vocabulary, second_version

from operator_embedding_shards import (
    ApplyOutcome,
    CheckpointError,
    EmbeddingShards,
    ParameterStore,
    ShardSpec,
    UnknownVocabularyVersion,
    VocabularyRegistry,
    build_vocabulary,
)


def build_setup(*, aggregation="sum", learning_rate=1.0):
    vocab = fixture_vocabulary()
    registry = VocabularyRegistry([vocab])
    op = EmbeddingShards(registry, aggregation=aggregation)
    return registry, op


class ApplyTest(unittest.TestCase):
    def test_applies_gradient_to_the_named_shard_only(self):
        registry, op = build_setup(learning_rate=0.5)
        store = ParameterStore(registry, learning_rate=0.5)
        forward = op.forward([5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(2.0, 4.0)])
        report = store.apply(batch)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.APPLIED])
        self.assertEqual(store.values("2026-11-a", 5), (1.0, 5.0))
        self.assertEqual(store.values("2026-11-a", 4), (3.0, 0.0))

    def test_repeated_ids_update_row_once_with_merged_gradient(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([5, 0, 5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 0.0), (0.0, 0.0), (2.0, 3.0)])
        report = store.apply(batch)
        self.assertEqual(len(report), 2)
        self.assertEqual(store.values("2026-11-a", 5), (3.0, 6.0))
        self.assertEqual(store.values("2026-11-a", 0), (1.0, 0.0))

    def test_reapplying_the_same_batch_is_idempotent(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])
        store.apply(batch)
        after_first = store.values("2026-11-a", 5)
        report = store.apply(batch)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.DUPLICATE])
        self.assertEqual(store.values("2026-11-a", 5), after_first)

    def test_disabled_ids_never_reach_the_store(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        before = store.values("2026-11-a", 6)
        forward = op.forward([6, 0], step=1, micro_batch_id="m")
        batch = op.backward(forward, [None, (1.0, 1.0)])
        store.apply(batch)
        self.assertEqual(store.values("2026-11-a", 6), before)
        self.assertEqual(len(batch.blocked), 1)

    def test_store_rejects_a_record_for_a_disabled_row(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([6], step=1, micro_batch_id="m")
        batch = op.backward(forward, [None])
        # A record cannot come from backward, so fabricate one to prove the
        # store is defensive even if a caller bypasses the operator.
        from operator_embedding_shards import UpdateRecord, UpdateTarget

        record = UpdateRecord(
            target=UpdateTarget("2026-11-a", 6, "b", 4, 2),
            gradient=(9.0, 9.0),
            occurrences=(0,),
            aggregation=batch.aggregation,
            layout_fingerprint=fixture_vocabulary().layout_fingerprint,
            row_fingerprint=fixture_vocabulary().row_fingerprint(6),
            micro_batch_id="m",
            step=1,
        )
        before = store.values("2026-11-a", 6)
        report = store.apply([record], step=1)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.DISABLED])
        self.assertEqual(store.values("2026-11-a", 6), before)

    def test_learning_rate_zero_leaves_rows_unchanged(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])
        store.apply(batch, learning_rate=0.0)
        self.assertEqual(store.values("2026-11-a", 5), (0.0, 3.0))

    def test_shard_view_groups_rows_by_shard(self):
        registry, _ = build_setup()
        store = ParameterStore(registry)
        view = store.shard_view("2026-11-a")
        self.assertEqual(set(view), {"a", "b"})
        self.assertEqual(view["b"][5], (0.0, 3.0))


class VersionSwitchTest(unittest.TestCase):
    def _mixed_setup(self):
        registry = VocabularyRegistry([fixture_vocabulary(), second_version()])
        op = EmbeddingShards(registry, default_version="2026-11-a")
        store = ParameterStore(registry)
        return registry, op, store

    def test_mid_batch_switch_keeps_each_position_on_its_own_version(self):
        _, op, store = self._mixed_setup()
        forward = op.forward(
            [0, 0],
            versions=["2026-11-a", "2026-11-b"],
            step=1,
            micro_batch_id="m",
        )
        self.assertEqual(forward[0].values, (1.0, 0.0))
        self.assertEqual(forward[1].values, (10.0, 10.0))
        self.assertEqual([vector.version for vector in forward], ["2026-11-a", "2026-11-b"])

    def test_vector_produced_under_old_version_is_not_overwritten_by_new(self):
        _, op, store = self._mixed_setup()
        forward_a = op.forward([0], versions="2026-11-a", step=1, micro_batch_id="a")
        batch_a = op.backward(forward_a, [(1.0, 1.0)])
        store.apply(batch_a)
        snapshot_a = store.values("2026-11-a", 0)

        forward_b = op.forward([0], versions="2026-11-b", step=2, micro_batch_id="b")
        batch_b = op.backward(forward_b, [(100.0, 100.0)])
        store.apply(batch_b)

        self.assertEqual(store.values("2026-11-a", 0), snapshot_a)
        self.assertEqual(store.values("2026-11-b", 0), (110.0, 110.0))

    def test_version_drop_makes_old_updates_unattributable(self):
        _, op, _ = self._mixed_setup()
        forward = op.forward([0], versions="2026-11-b", step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])

        only_a = VocabularyRegistry([fixture_vocabulary()])
        store = ParameterStore(only_a)
        report = store.apply(batch)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.UNATTRIBUTED])
        self.assertIn("no longer loaded", report.results[0].reason)

    def test_changed_layout_marks_update_stale(self):
        vocab = fixture_vocabulary()
        store = ParameterStore(VocabularyRegistry([vocab]))
        op = EmbeddingShards(VocabularyRegistry([vocab]))
        forward = op.forward([5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])

        moved = build_vocabulary(
            "2026-11-a",
            [
                ShardSpec("a", 0, ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0))),
                ShardSpec("b", 5, ((3.0, 0.0), (0.0, 3.0), (2.0, 2.0))),
            ],
            unknown_id=-1,
            disabled=frozenset({6}),
        )
        # Replace the layout in a fresh store sharing the same version name.
        store._states["2026-11-a"] = type(store._states["2026-11-a"]).from_vocabulary(moved)
        report = store.apply(batch)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.STALE])

    def test_advanced_row_marks_old_update_stale(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([5], step=1, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])
        # A concurrent step advances the row between forward and apply.
        store.apply(
            [
                record.__class__(
                    target=record.target,
                    gradient=(50.0, 50.0),
                    occurrences=record.occurrences,
                    aggregation=record.aggregation,
                    layout_fingerprint=record.layout_fingerprint,
                    row_fingerprint=record.row_fingerprint,
                    micro_batch_id="other",
                    step=1,
                )
                for record in batch.records
            ]
        )
        report = store.apply(batch, step=1)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.STALE])

    def test_step_mismatch_is_rejected(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        forward = op.forward([5], step=4, micro_batch_id="m")
        batch = op.backward(forward, [(1.0, 1.0)])
        report = store.apply(batch, step=5)
        self.assertEqual([item.outcome for item in report], [ApplyOutcome.STEP_MISMATCH])


class CheckpointAndResumeTest(unittest.TestCase):
    def _train_one_step(self, store, op, ids, gradients, step):
        forward = op.forward(ids, step=step, micro_batch_id=f"mb-{step}")
        batch = op.backward(forward, gradients)
        store.apply(batch)
        return batch

    def test_checkpoint_round_trip_preserves_values_and_ledger(self):
        registry, op = build_setup(learning_rate=0.5)
        store = ParameterStore(registry, learning_rate=0.5)
        self._train_one_step(store, op, [5], [(2.0, 2.0)], step=1)
        payload = store.checkpoint()

        restored = ParameterStore.from_checkpoint(registry, payload, learning_rate=0.5)
        self.assertEqual(restored.values("2026-11-a", 5), store.values("2026-11-a", 5))
        self.assertEqual(restored.applied_update_ids(), store.applied_update_ids())

    def test_resume_distinguishes_replayable_from_lost_updates(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        applied = self._train_one_step(store, op, [0, 5], [(1.0, 0.0), (0.0, 1.0)], step=1)
        payload = store.checkpoint()

        pending_forward = op.forward([0, 5], step=2, micro_batch_id="pending")
        pending = op.backward(pending_forward, [(1.0, 1.0), (1.0, 1.0)])

        resumed = ParameterStore.from_checkpoint(registry, payload)
        replay = resumed.replayability(applied)
        self.assertEqual(len(replay.replayable), 2)
        self.assertEqual(replay.lost, ())

        pending_replay = resumed.replayability(pending)
        self.assertEqual(len(pending_replay.replayable), 2)
        self.assertEqual(pending_replay.lost, ())
        # Dry-run must not mutate state.
        self.assertEqual(
            resumed.values("2026-11-a", 5), resumed.vocabulary("2026-11-a").lookup(5)
        )

    def test_lost_updates_are_reported_after_row_advanced(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        # A forward is taken, then another step advances the same row before
        # this batch reaches the optimiser.
        stale_forward = op.forward([5], step=1, micro_batch_id="stale")
        stale_batch = op.backward(stale_forward, [(1.0, 1.0)])

        fresh_forward = op.forward([5], step=1, micro_batch_id="fresh")
        fresh_batch = op.backward(fresh_forward, [(1.0, 1.0)])
        store.apply(fresh_batch)

        report = store.replayability(stale_batch)
        self.assertEqual(report.replayable, ())
        self.assertEqual(len(report.lost), 1)
        self.assertEqual(report.lost_details[0].outcome, ApplyOutcome.STALE)

    def test_checkpoint_rejects_unknown_version(self):
        registry, op = build_setup()
        store = ParameterStore(registry)
        payload = store.checkpoint()
        payload["versions"]["ghost"] = payload["versions"].pop("2026-11-a")
        with self.assertRaises(UnknownVocabularyVersion):
            ParameterStore.from_checkpoint(registry, payload)

    def test_checkpoint_rejects_layout_mismatch(self):
        registry, _ = build_setup()
        store = ParameterStore(registry)
        payload = store.checkpoint()
        payload["versions"]["2026-11-a"]["layout_fingerprint"] = "wrong"
        with self.assertRaises(CheckpointError):
            ParameterStore.from_checkpoint(registry, payload)

    def test_checkpoint_rejects_row_outside_layout(self):
        registry, _ = build_setup()
        store = ParameterStore(registry)
        payload = store.checkpoint()
        payload["versions"]["2026-11-a"]["rows"]["999"] = [0.0, 0.0]
        with self.assertRaises(CheckpointError):
            ParameterStore.from_checkpoint(registry, payload)

    def test_checkpoint_rejects_bad_format(self):
        registry, _ = build_setup()
        with self.assertRaises(CheckpointError):
            ParameterStore.from_checkpoint(registry, {"format": "nope"})


if __name__ == "__main__":
    unittest.main()
