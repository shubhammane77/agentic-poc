import unittest

import tests._path_setup  # noqa: F401

from agentic_testgen.logging import SecretRedactor


class SecretRedactorTests(unittest.TestCase):
    def test_redacts_known_secrets_and_patterns(self) -> None:
        redactor = SecretRedactor(["topsecret"])
        text = "token=topsecret and api=gsk_ABC1234567890"
        redacted = redactor.redact(text)
        self.assertNotIn("topsecret", redacted)
        self.assertNotIn("gsk_ABC1234567890", redacted)
        self.assertIn("[REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()
