import unittest
from pathlib import Path

from qlora_customer_support.data import (
    format_example,
    format_prompt,
    load_jsonl,
    mask_prompt_labels,
    train_eval_split,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "support.jsonl"


class DataTests(unittest.TestCase):
    def test_corpus_has_customer_support_pairs(self):
        rows = load_jsonl(DATA)
        self.assertGreaterEqual(len(rows), 24)
        for row in rows:
            self.assertTrue(row["instruction"].startswith("Customer:"))
            self.assertIn("Northline", row["response"])

    def test_prompt_is_a_prefix_of_the_full_example(self):
        prompt, full = format_example("Customer: Where is my order?", "The tracking link is in the shipment email.")
        self.assertEqual(prompt, format_prompt("Customer: Where is my order?"))
        self.assertTrue(full.startswith(prompt))
        self.assertTrue(full.endswith("shipment email."))

    def test_loss_is_masked_off_the_prompt(self):
        labels = mask_prompt_labels([11, 12, 13, 14, 15], prompt_len=3)
        self.assertEqual(labels, [-100, -100, -100, 14, 15])

    def test_mask_keeps_at_least_one_reply_token(self):
        labels = mask_prompt_labels([1, 2, 3], prompt_len=99)
        self.assertEqual(labels, [-100, -100, 3])

    def test_split_is_deterministic_and_disjoint(self):
        rows = load_jsonl(DATA)
        train_a, eval_a = train_eval_split(rows, eval_size=4, seed=7)
        train_b, eval_b = train_eval_split(rows, eval_size=4, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(len(eval_a), 4)
        self.assertEqual(len(train_a) + len(eval_a), len(rows))
        train_keys = {row["instruction"] for row in train_a}
        self.assertTrue(all(row["instruction"] not in train_keys for row in eval_a))


if __name__ == "__main__":
    unittest.main()
