from __future__ import annotations

import unittest

from cli import build_parser


class CliTests(unittest.TestCase):
    def test_parser_builds_and_formats_help(self) -> None:
        parser = build_parser()

        help_text = parser.format_help()

        self.assertIn("--transaction-cost", help_text)
        self.assertIn("0.1%", help_text)


if __name__ == "__main__":
    unittest.main()
