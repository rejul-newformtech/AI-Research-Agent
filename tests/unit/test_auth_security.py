"""Unit tests for authentication security utilities (JWT and password hashing)."""

import unittest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestSecurityUtilities(unittest.TestCase):
    """Test cryptographic hashing and JWT token operations."""

    def test_password_hashing(self):
        plain_password = "SuperSecretPassword123!"
        hashed = hash_password(plain_password)
        self.assertNotEqual(plain_password, hashed)
        self.assertTrue(verify_password(plain_password, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_jwt_token_flow(self):
        data = {"sub": "testuser", "user_id": 42, "role": "researcher"}
        token = create_access_token(data)
        self.assertIsInstance(token, str)

        payload = decode_access_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "testuser")
        self.assertEqual(payload["user_id"], 42)
        self.assertEqual(payload["role"], "researcher")


if __name__ == "__main__":
    unittest.main()
