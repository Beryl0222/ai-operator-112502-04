"""Vocabulary layout, resolution and version registry tests."""

from __future__ import annotations

import unittest

from helpers import fixture_vocabulary, second_version, simple_vocabulary

from operator_embedding_shards import (
    LayoutConflictError,
    ShardLayoutError,
    ShardSpec,
    TokenStatus,
    UnknownVocabularyVersion,
    VocabularyError,
    VocabularyRegistry,
    VocabularyVersion,
    build_vocabulary,
)


class ShardGeometryTest(unittest.TestCase):
    def test_shard_reports_range_and_rows(self):
        shard = ShardSpec("b", 4, ((3.0, 0.0), (0.0, 3.0)))
        self.assertEqual(shard.start, 4)
        self.assertEqual(shard.stop, 6)
        self.assertEqual(shard.length, 2)
        self.assertEqual(shard.dim, 2)
        self.assertTrue(shard.contains(5))
        self.assertFalse(shard.contains(6))
        self.assertEqual(shard.row(5), (0.0, 3.0))

    def test_shard_rejects_inconsistent_row_dimensions(self):
        with self.assertRaises(ShardLayoutError):
            ShardSpec("a", 0, ((1.0, 2.0), (3.0,)))

    def test_shard_rejects_empty_values(self):
        with self.assertRaises(ShardLayoutError):
            ShardSpec("a", 0, ())

    def test_shard_rejects_negative_start(self):
        with self.assertRaises(ShardLayoutError):
            ShardSpec("a", -2, ((1.0,),))


class VocabularyLayoutTest(unittest.TestCase):
    def test_fixture_resolves_every_id_status(self):
        vocab = fixture_vocabulary()
        self.assertEqual(vocab.dim, 2)
        self.assertEqual(vocab.shard_ids, ("a", "b"))
        ok = vocab.resolve(5)
        self.assertIs(ok.status, TokenStatus.OK)
        self.assertEqual((ok.shard_id, ok.shard_start, ok.local_index), ("b", 4, 1))
        self.assertEqual(vocab.vector_for(5)[0], (0.0, 3.0))
        self.assertIs(vocab.resolve(-1).status, TokenStatus.UNKNOWN)
        self.assertIs(vocab.resolve(6).status, TokenStatus.DISABLED)
        self.assertIs(vocab.resolve(9).status, TokenStatus.MISSING)
        self.assertIsNone(vocab.resolve(9).shard_id)

    def test_disabled_id_keeps_shard_attribution_but_blocks_update(self):
        vocab = fixture_vocabulary()
        resolution = vocab.resolve(6)
        self.assertEqual(resolution.shard_id, "b")
        self.assertFalse(resolution.allows_update)
        self.assertFalse(resolution.is_embeddable)

    def test_gap_between_shards_is_missing_not_unknown(self):
        vocab = build_vocabulary(
            "gaps",
            [ShardSpec("a", 0, ((0.0,),)), ShardSpec("b", 5, ((5.0,),))],
        )
        for token_id in (1, 2, 3, 4):
            resolution = vocab.resolve(token_id)
            self.assertIs(resolution.status, TokenStatus.MISSING)
            self.assertIn("gap", resolution.reason)
        self.assertIs(vocab.resolve(6).status, TokenStatus.MISSING)

    def test_unknown_fallback_vector_is_configurable(self):
        vocab = simple_vocabulary(unknown_vector=(7.0, 8.0))
        values, resolution = vocab.vector_for(-1)
        self.assertIs(resolution.status, TokenStatus.UNKNOWN)
        self.assertEqual(values, (7.0, 8.0))

    def test_unknown_fallback_defaults_to_zero_vector(self):
        vocab = simple_vocabulary()
        values, _ = vocab.vector_for(-1)
        self.assertEqual(values, (0.0, 0.0))

    def test_disabled_fallback_is_separate_from_unknown_fallback(self):
        vocab = simple_vocabulary(
            disabled=frozenset({4}),
            unknown_vector=(1.0, 1.0),
            disabled_vector=(2.0, 2.0),
        )
        unknown_values, _ = vocab.vector_for(-1)
        disabled_values, disabled_resolution = vocab.vector_for(4)
        self.assertEqual(unknown_values, (1.0, 1.0))
        self.assertEqual(disabled_values, (2.0, 2.0))
        self.assertIs(disabled_resolution.status, TokenStatus.DISABLED)

    def test_unknown_id_may_not_also_be_disabled(self):
        with self.assertRaises(VocabularyError):
            simple_vocabulary(unknown_id=-1, disabled=frozenset({-1}))

    def test_overlapping_shards_are_rejected(self):
        with self.assertRaises(ShardLayoutError):
            VocabularyVersion(
                "overlap",
                (ShardSpec("a", 0, ((1.0,), (2.0,))), ShardSpec("b", 1, ((3.0,),))),
            )

    def test_duplicate_shard_ids_are_rejected(self):
        with self.assertRaises(ShardLayoutError):
            VocabularyVersion(
                "dup",
                (ShardSpec("a", 0, ((1.0,),)), ShardSpec("a", 4, ((2.0,),))),
            )

    def test_mismatched_shard_dimensions_are_rejected(self):
        with self.assertRaises(ShardLayoutError):
            VocabularyVersion(
                "dim",
                (ShardSpec("a", 0, ((1.0,),)), ShardSpec("b", 4, ((2.0, 3.0),))),
            )

    def test_touching_shards_remain_distinct(self):
        vocab = fixture_vocabulary()
        self.assertEqual(vocab.resolve(3).shard_id, "a")
        self.assertEqual(vocab.resolve(4).shard_id, "b")

    def test_row_fingerprint_tracks_values(self):
        vocab = simple_vocabulary()
        before = vocab.row_fingerprint(3)
        updated = vocab.with_values({3: (99.0, 99.0)})
        self.assertNotEqual(before, updated.row_fingerprint(3))
        self.assertEqual(vocab.layout_fingerprint, updated.layout_fingerprint)

    def test_layout_fingerprint_ignores_row_values_but_not_geometry(self):
        vocab = simple_vocabulary("v1")
        same_shape = build_vocabulary(
            "v1", [ShardSpec("s", 0, tuple((9.0, 9.0) for _ in range(6)))]
        )
        moved = build_vocabulary("v1", [ShardSpec("s", 1, tuple((0.0, 0.0) for _ in range(6)))])
        self.assertEqual(vocab.layout_fingerprint, same_shape.layout_fingerprint)
        self.assertNotEqual(vocab.layout_fingerprint, moved.layout_fingerprint)

    def test_resolve_rejects_non_integer_ids(self):
        vocab = simple_vocabulary()
        with self.assertRaises(VocabularyError):
            vocab.resolve("3")  # type: ignore[arg-type]
        with self.assertRaises(VocabularyError):
            vocab.resolve(True)  # type: ignore[arg-type]


class RegistryTest(unittest.TestCase):
    def test_registry_holds_multiple_versions(self):
        registry = VocabularyRegistry([simple_vocabulary("v1"), second_version()])
        self.assertEqual(sorted(registry.versions), ["2026-11-b", "v1"])
        self.assertIn("v1", registry)
        self.assertEqual(registry.get("v1").version, "v1")

    def test_unknown_version_raises_with_known_versions(self):
        registry = VocabularyRegistry([simple_vocabulary("v1")])
        with self.assertRaises(UnknownVocabularyVersion) as caught:
            registry.get("nope")
        self.assertIn("v1", str(caught.exception))

    def test_registering_identical_layout_is_idempotent(self):
        registry = VocabularyRegistry()
        first = registry.register(simple_vocabulary("v1"))
        second = registry.register(simple_vocabulary("v1"))
        self.assertIs(first, second)
        self.assertEqual(len(registry), 1)

    def test_registering_conflicting_layout_is_rejected(self):
        registry = VocabularyRegistry([simple_vocabulary("v1")])
        conflicting = build_vocabulary(
            "v1", [ShardSpec("s", 0, tuple((1.0, 1.0) for _ in range(6)))]
        )
        with self.assertRaises(LayoutConflictError):
            registry.register(conflicting)

    def test_replace_updates_rows_only_for_same_layout(self):
        registry = VocabularyRegistry([simple_vocabulary("v1")])
        registry.replace(simple_vocabulary("v1").with_values({0: (5.0, 5.0)}))
        self.assertEqual(registry.get("v1").lookup(0), (5.0, 5.0))
        with self.assertRaises(LayoutConflictError):
            registry.replace(
                build_vocabulary(
                    "v1", [ShardSpec("s", 8, tuple((0.0, 0.0) for _ in range(6)))]
                )
            )


if __name__ == "__main__":
    unittest.main()
