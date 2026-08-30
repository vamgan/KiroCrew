"""The ``mobile_connect`` CPP seam and its governance chokepoints.

Four surfaces under pin:

1. the Default provider reproduces the personal-install pair (tailnet QR +
   one-time login link) — the seam's "byte-identical standalone" contract;
2. ``_governed_methods`` consults ``capabilities.mobile_connect`` per id and
   drops denied/malformed rows (and hides everything when the capability is
   off or the seam read degrades);
3. the listing endpoint's auth floor (unauthenticated 401, app token 403);
4. the mint endpoints re-run the same decision — the filtered list is
   presentation, never the control — pinned at the wire on
   ``api_auth_mobile_link``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from kiro_crew.dashboard.handlers import mobile_connect
from kiro_crew.platform.defaults import DefaultMobileConnectProvider
from kiro_crew.platform.governance import SCOPE_CATALOG
from kiro_crew.platform.interfaces import MobileConnectMethod


@dataclass
class _Decision:
    permitted: bool
    reason: str = ""


def _permits(denied: set[str]):
    """A governance_permits stand-in denying exactly the given items."""

    def fake(scope, item, **kwargs):
        assert scope == mobile_connect.MOBILE_CONNECT_SCOPE
        return _Decision(permitted=item not in denied, reason=f"denied {item!r}")

    return fake


class TestDefaultProvider:
    def test_personal_install_pair(self) -> None:
        methods = DefaultMobileConnectProvider().connect_methods()
        assert [(m.id, m.kind) for m in methods] == [
            ("tailnet-qr", "tailnet_qr"),
            ("login-link", "login_link"),
        ]

    def test_scope_catalog_row_is_a_scoped_capability(self) -> None:
        spec = SCOPE_CATALOG["capabilities.mobile_connect"]
        # Default True keeps the standalone pair working; the ``methods``
        # ruleset is what lets a policy narrow per id.
        assert spec.capability_default is True
        assert spec.scope_matchers == {"methods": "identifier"}


class TestGovernedMethods:
    def _with_provider(self, methods, denied: set[str] = frozenset()):
        ctx = MagicMock()
        ctx.mobile_connect.connect_methods.return_value = methods
        with (
            patch("kiro_crew.platform.context.current_context", return_value=ctx),
            patch(
                "kiro_crew.platform.governance_profiles.governance_permits",
                side_effect=_permits(denied),
            ),
        ):
            return mobile_connect._governed_methods()

    def test_all_permitted(self) -> None:
        out = self._with_provider(DefaultMobileConnectProvider().connect_methods())
        assert out == [
            {"id": "tailnet-qr", "kind": "tailnet_qr"},
            {"id": "login-link", "kind": "login_link"},
        ]

    def test_capability_off_hides_everything(self) -> None:
        out = self._with_provider(DefaultMobileConnectProvider().connect_methods(), denied={""})
        assert out == []

    def test_methods_ruleset_narrows_per_id(self) -> None:
        # The per-id check must pass the ``methods:<id>`` item — a mutation
        # that passes "" for every row keeps both and fails this test.
        out = self._with_provider(
            DefaultMobileConnectProvider().connect_methods(),
            denied={"methods:tailnet-qr"},
        )
        assert out == [{"id": "login-link", "kind": "login_link"}]

    def test_malformed_descriptor_dropped(self) -> None:
        out = self._with_provider(
            [MobileConnectMethod(id="", kind="x"), MobileConnectMethod(id="ok", kind="")]
        )
        assert out == []

    def test_degraded_seam_read_hides_the_entry(self) -> None:
        ctx = MagicMock()
        ctx.mobile_connect.connect_methods.side_effect = RuntimeError("adapter broke")
        with patch("kiro_crew.platform.context.current_context", return_value=ctx):
            assert mobile_connect._governed_methods() == []


def _request(*, user: str = "alice", app: str = "") -> MagicMock:
    request = MagicMock(spec=web.Request)
    request.get.side_effect = {"user": user, "app": app}.get
    request.headers = {}
    return request


def _call_methods(request: MagicMock):
    with patch("kiro_crew.dashboard.handlers.mobile_connect.check_origin", return_value=True):
        return asyncio.run(mobile_connect.api_mobile_connect_methods(request))


class TestMethodsEndpoint:
    def test_unauthenticated_401(self) -> None:
        resp = _call_methods(_request(user=""))
        assert resp.status == 401

    def test_app_token_403(self) -> None:
        resp = _call_methods(_request(app="some-app"))
        assert resp.status == 403

    def test_ok_lists_governed_methods(self) -> None:
        rows = [{"id": "login-link", "kind": "login_link"}]
        with patch.object(mobile_connect, "_governed_methods", return_value=rows):
            resp = _call_methods(_request())
        assert resp.status == 200
        body = json.loads(resp.text)
        assert body == {"enabled": True, "methods": rows}

    def test_empty_methods_reports_disabled(self) -> None:
        with patch.object(mobile_connect, "_governed_methods", return_value=[]):
            resp = _call_methods(_request())
        assert json.loads(resp.text) == {"enabled": False, "methods": []}


class TestMintDeniedReason:
    def test_permitted_is_empty(self) -> None:
        with patch(
            "kiro_crew.platform.governance_profiles.governance_permits",
            side_effect=_permits(set()),
        ):
            assert mobile_connect.mint_denied_reason("login-link") == ""

    @pytest.mark.parametrize("denied", [{""}, {"methods:login-link"}])
    def test_either_half_denies(self, denied: set[str]) -> None:
        with patch(
            "kiro_crew.platform.governance_profiles.governance_permits",
            side_effect=_permits(denied),
        ):
            assert mobile_connect.mint_denied_reason("login-link") != ""


class TestMintEndpointsReCheck:
    def test_mobile_link_mint_refused_when_method_denied(self) -> None:
        """The wire-level pin: a governance denial 403s BEFORE any minting."""
        from kiro_crew.dashboard.handlers import auth_mobile

        request = MagicMock(spec=web.Request)
        request.get.side_effect = {"user": "alice", "app": ""}.get
        request.headers = {"Origin": ""}
        with (
            patch("kiro_crew.dashboard.handlers.auth_mobile.check_origin", return_value=True),
            patch(
                "kiro_crew.dashboard.handlers.auth_mobile.mint_denied_reason",
                return_value="mobile connect disabled by policy",
            ),
        ):
            resp = asyncio.run(auth_mobile.api_auth_mobile_link(request))
        assert resp.status == 403
        assert json.loads(resp.text)["code"] == "governance_denied"

    def test_tailnet_qr_mint_refused_when_method_denied(self) -> None:
        from kiro_crew.dashboard.handlers import tailnet_mobile

        request = MagicMock(spec=web.Request)
        request.app = {}
        with (
            patch.object(tailnet_mobile, "_guard", return_value=None),
            patch(
                "kiro_crew.dashboard.handlers.tailnet_mobile.mint_denied_reason",
                return_value="mobile connect disabled by policy",
            ),
            patch.object(tailnet_mobile, "_audit_async") as audit,
        ):
            resp = asyncio.run(tailnet_mobile.api_tailnet_mobile_qr(request))
        assert resp.status == 403
        assert json.loads(resp.text)["code"] == "governance_denied"
        # The denial is audited, mirroring every other refusal in this handler.
        assert audit.await_count == 1
