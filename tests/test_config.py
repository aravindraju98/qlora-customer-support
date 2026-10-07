import unittest

import torch
from peft import TaskType

from qlora_customer_support.qlora_config import TARGET_MODULES, bitsandbytes_config, lora_config


class QLoRAConfigTests(unittest.TestCase):
    def test_quantization_is_4bit_nf4_with_double_quant(self):
        config = bitsandbytes_config()
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(config.bnb_4bit_use_double_quant)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)

    def test_lora_adapters_target_attention_projections(self):
        config = lora_config()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertAlmostEqual(config.lora_dropout, 0.05)
        self.assertEqual(set(config.target_modules), set(TARGET_MODULES))
        self.assertEqual(config.task_type, TaskType.CAUSAL_LM)
        self.assertFalse(config.inference_mode)


if __name__ == "__main__":
    unittest.main()
