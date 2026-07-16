"""Backend authorization tests for the optional admin/viewer access mode.

api.py's require_admin() gates upload/reprocess/delete/clear/eval-run/
feedback-resolve/drive-config -- these tests exercise it directly rather
than through the full FastAPI app (no live services needed), covering both
ACCESS_MODE=open (today's default, must be unaffected) and admin_viewer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import api


def _call_require_admin(x_api_key=None, vault_admin_session=None):
    return asyncio.run(
        api.require_admin(x_api_key=x_api_key, vault_admin_session=vault_admin_session)
    )


class TestOpenMode:
    """ACCESS_MODE=open must behave exactly like the pre-existing API_KEY-only
    gate -- this feature must not change default/dev behavior."""

    def test_no_api_key_configured_allows_everyone(self):
        with (
            patch.object(api, "ACCESS_MODE", "open"),
            patch.object(api, "API_KEY", ""),
        ):
            _call_require_admin()  # must not raise

    def test_api_key_configured_requires_matching_header(self):
        with (
            patch.object(api, "ACCESS_MODE", "open"),
            patch.object(api, "API_KEY", "secret123"),
        ):
            with pytest.raises(HTTPException) as exc:
                _call_require_admin(x_api_key=None)
            assert exc.value.status_code == 401

    def test_api_key_configured_passes_with_matching_header(self):
        with (
            patch.object(api, "ACCESS_MODE", "open"),
            patch.object(api, "API_KEY", "secret123"),
        ):
            _call_require_admin(x_api_key="secret123")  # must not raise


class TestAdminViewerMode:
    def test_no_credentials_is_forbidden(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", ""),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            with pytest.raises(HTTPException) as exc:
                _call_require_admin()
            assert exc.value.status_code == 403

    def test_valid_api_key_header_is_allowed(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", "secret123"),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            _call_require_admin(x_api_key="secret123")  # must not raise

    def test_valid_session_cookie_is_allowed(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", ""),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            token = api._admin_session_token()
            _call_require_admin(vault_admin_session=token)  # must not raise

    def test_forged_session_cookie_is_rejected(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", ""),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            with pytest.raises(HTTPException) as exc:
                _call_require_admin(vault_admin_session="not-the-real-token")
            assert exc.value.status_code == 403

    def test_session_token_from_a_different_secret_is_rejected(self):
        """A cookie signed under a different SESSION_SECRET (e.g. before a
        secret rotation) must not validate."""
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", ""),
            patch.object(api, "SESSION_SECRET", "secret-a"),
        ):
            token = api._admin_session_token()
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "API_KEY", ""),
            patch.object(api, "SESSION_SECRET", "secret-b"),
        ):
            with pytest.raises(HTTPException) as exc:
                _call_require_admin(vault_admin_session=token)
            assert exc.value.status_code == 403


class TestAdminLoginEndpoint:
    def test_login_rejected_when_not_in_admin_viewer_mode(self):
        with patch.object(api, "ACCESS_MODE", "open"):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(
                    api.admin_login(
                        api.AdminLoginRequest(password="anything"),
                        response=api.Response(),
                    )
                )
            assert exc.value.status_code == 400

    def test_login_rejects_wrong_password(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "ADMIN_PASSWORD", "correct-horse"),
        ):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(
                    api.admin_login(
                        api.AdminLoginRequest(password="wrong"),
                        response=api.Response(),
                    )
                )
            assert exc.value.status_code == 401

    def test_login_accepts_correct_password_and_sets_cookie(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "ADMIN_PASSWORD", "correct-horse"),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            response = api.Response()
            result = asyncio.run(
                api.admin_login(
                    api.AdminLoginRequest(password="correct-horse"), response=response
                )
            )
            assert result == {"status": "ok"}
            assert "vault_admin_session" in response.headers.get("set-cookie", "")


class TestAdminSessionEndpoint:
    def test_open_mode_reports_is_admin_true(self):
        """In open mode everyone effectively has admin capability -- the
        frontend must not hide admin UI behind a login that doesn't exist."""
        with patch.object(api, "ACCESS_MODE", "open"):
            result = asyncio.run(api.admin_session(vault_admin_session=None))
        assert result == {"access_mode": "open", "is_admin": True}

    def test_admin_viewer_mode_reports_is_admin_false_without_cookie(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            result = asyncio.run(api.admin_session(vault_admin_session=None))
        assert result == {"access_mode": "admin_viewer", "is_admin": False}

    def test_admin_viewer_mode_reports_is_admin_true_with_valid_cookie(self):
        with (
            patch.object(api, "ACCESS_MODE", "admin_viewer"),
            patch.object(api, "SESSION_SECRET", "test-secret"),
        ):
            token = api._admin_session_token()
            result = asyncio.run(api.admin_session(vault_admin_session=token))
        assert result == {"access_mode": "admin_viewer", "is_admin": True}
