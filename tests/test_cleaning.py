"""Unit tests for the parsing / cleaning / payload logic (no network needed).

Run from the repo root:   python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pubmed_summarizer as ps  # noqa: E402


def clean(text):
    return ps.clean_all(ps.load_raw_entries(text))


class CleaningTests(unittest.TestCase):
    def test_sample_data(self):
        records, skipped = clean((ROOT / "sample_data.json").read_text(encoding="utf-8"))
        self.assertEqual([r.record_id for r in records], ["PMID:349281", "PMID:8829102"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["id"], "ERR_NOT_FOUND")

    def test_null_title_and_unknown_authors(self):
        records, _ = clean((ROOT / "sample_data.json").read_text(encoding="utf-8"))
        self.assertIsNone(records[1].title)
        self.assertEqual(records[1].authors, [])
        self.assertEqual(records[0].authors, ["Smith, J", "Doe, A"])

    def test_inequality_signs_survive(self):
        text = json.dumps([{"id": 1, "abstract": "Response was higher (p<0.05) and HR>1.2 in the treated arm."}])
        records, _ = clean(text)
        self.assertIn("p<0.05", records[0].abstract)
        self.assertIn("HR>1.2", records[0].abstract)

    def test_xml_tags_stripped(self):
        text = json.dumps([{"id": 2, "abstract": '<AbstractText Label="RESULTS">Mortality fell by <i>12</i>%.</AbstractText>'}])
        records, _ = clean(text)
        self.assertEqual(records[0].abstract, "Mortality fell by 12 %.")

    def test_trailing_commas_repaired(self):
        records, skipped = clean('[{"id": 5, "abstract": "A long enough abstract about outcomes.",},]')
        self.assertEqual(len(records), 1)
        self.assertEqual(skipped, [])

    def test_json_lines_fallback(self):
        text = (
            '{"id": 1, "abstract": "First abstract about heart failure outcomes."}\n'
            '{"id": 2, "abstract": "Second abstract about renal outcomes."}'
        )
        records, _ = clean(text)
        self.assertEqual([r.pmid for r in records], ["1", "2"])

    def test_non_object_entries_skipped(self):
        records, skipped = clean('[1, "x", null, {"id": 7, "abstract": "A usable abstract about trial results."}]')
        self.assertEqual(len(records), 1)
        self.assertEqual(len(skipped), 3)

    def test_duplicate_pmid_skipped(self):
        text = json.dumps([
            {"id": "PMID:9", "abstract": "First copy of this abstract text."},
            {"id": 9, "abstract": "Second copy of this abstract text."},
        ])
        records, skipped = clean(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(skipped[0]["reason"], "duplicate PMID")

    def test_invalid_id_with_abstract_is_kept(self):
        records, _ = clean('[{"id": "ERR_NOT_FOUND", "abstract": "This abstract is fine despite the bad id."}]')
        self.assertEqual(records[0].record_id, "record_1")
        self.assertIsNone(records[0].pmid)

    def test_short_abstract_skipped(self):
        records, skipped = clean('[{"id": 3, "abstract": "n/a"}, {"id": 4, "abstract": "ok"}]')
        self.assertEqual(records, [])
        self.assertEqual(len(skipped), 2)

    def test_long_abstract_truncated(self):
        records, _ = clean(json.dumps([{"id": 8, "abstract": "word " * 3000}]))
        self.assertTrue(records[0].abstract_truncated)
        self.assertTrue(records[0].abstract.endswith("[...]"))
        self.assertLessEqual(len(records[0].abstract), ps.MAX_ABSTRACT_CHARS + 6)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            ps.load_raw_entries("   ")

    def test_garbage_input_raises(self):
        with self.assertRaises(ValueError):
            ps.load_raw_entries("this is not json at all")


class PayloadTests(unittest.TestCase):
    def test_payload_shape(self):
        records, _ = clean((ROOT / "sample_data.json").read_text(encoding="utf-8"))
        payload = ps.build_payload(records[0], "Llama-3.3-70B-Instruct")
        self.assertEqual(payload["model"], "Llama-3.3-70B-Instruct")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0.0)
        user = payload["messages"][1]["content"]
        self.assertIn("Summarize the clinical outcome of this study in one sentence.", user)
        self.assertIn("We tested 500 patients.", user)

    def test_missing_title_placeholder(self):
        records, _ = clean((ROOT / "sample_data.json").read_text(encoding="utf-8"))
        user = ps.build_payload(records[1], "m")["messages"][1]["content"]
        self.assertIn("Title: (not provided)", user)


if __name__ == "__main__":
    unittest.main()
