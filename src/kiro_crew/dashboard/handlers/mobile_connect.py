"""Phone-connection method listing — the CPP ``mobile_connect`` seam's consumer.

``GET /api/mobile-connect/methods`` answers ONE question for the dashboard:
which ways of handing a phone a live session exist on this deployment, under
the current governance ceiling. The rows come from
``current_context().mobile_connect`` (the personal-install Default is the
tailnet QR + one-time login link pair; an enterprise companion swaps the list)
and each id is filtered through the ``capabilities.mobile_connect`` scope.

The response is deliberately descriptor-only (``{id, kind}``): minting the
actual credential stays on each method's own endpoint (``/api/tailnet/mobile/qr``,
``/api/auth/mobile-link``), which re-run this same governance decision before
acting — a filtered list is presentation, never the control. An empty list
(edition returned none, policy denied all, or the seam read degraded) makes the
dashboard hide its "Connect your phone" entry rather than render dead buttons.

Auth floor matches ``/api/auth/mobile-link``'s read half: an authenticated,
non-app dashboard user. Restricted sessions may READ the list (the entry hides
nothing secret — kind names only); their mint attempts are refused by the mint
endpoints' own guards, which is where that refusal already lives.
"""

from __future__ import annotations

import logging

from aiohttp import web

from kiro_crew.dashboard.origin import check_origin

logger = logging.getLogger(__name__)

#: Governed scope for phone-connection methods (SCOPE_CATALOG row).
MOBILE_CONNECT_SCOPE = "capabilities.mobile_connect"


def _governed_methods() -> list[dict[str, str]]:
    """Seam read + governance filter, shared by the endpoint and future callers.

    ``safe_context_call`` fallback is ``[]``: a degraded seam read HIDES the
    entry instead of guessing at methods whose mint endpoints would then 403.
    The capability check is fail-closed (a wrong-permit widens an auth
    surface); a per-id denial drops that row and keeps the rest.
    """
    from kiro_crew.platform.context import current_context, safe_context_call
    from kiro_crew.platform.governance_profiles import governance_permits
    from kiro_crew.platform.interfaces import MobileConnectMethod

    methods: list[MobileConnectMethod] = safe_context_call(
        lambda: list(current_context().mobile_connect.connect_methods()),
        fallback_factory=list,
        log_message="mobile_connect.connect_methods degraded; hiding the connect entry",
    )
    if not methods:
        return []
    gate = governance_permits(MOBILE_CONNECT_SCOPE, "", log_warning=False, fail_closed=True)
    if not getattr(gate, "permitted", False):
        return []
    out: list[dict[str, str]] = []
    for m in methods:
        mid = getattr(m, "id", "")
        kind = getattr(m, "kind", "")
        if not mid or not kind:
            continue  # malformed descriptor: drop, never shadow
        scoped = governance_permits(
            MOBILE_CONNECT_SCOPE, f"methods:{mid}", log_warning=False, fail_closed=True
        )
        if getattr(scoped, "permitted", False):
            out.append({"id": mid, "kind": kind})
    return out


async def api_mobile_connect_methods(request: web.Request) -> web.Response:
    """GET /api/mobile-connect/methods → ``{"enabled": bool, "methods": [...]}``."""
    if not check_origin(request, require=False):
        return web.json_response({"error": "bad origin", "code": "bad_origin"}, status=403)
    if not request.get("user"):
        return web.json_response(
            {"error": "unauthenticated", "code": "unauthenticated"}, status=401
        )
    if request.get("app"):
        # App tokens act for an app, not the operator; connection methods are
        # an operator surface (mirrors /api/auth/mobile-link's refusal).
        return web.json_response(
            {"error": "app tokens cannot list connect methods", "code": "app_token_forbidden"},
            status=403,
        )
    methods = _governed_methods()
    return web.json_response({"enabled": bool(methods), "methods": methods})


def mint_denied_reason(method_id: str) -> str:
    """Governance re-check for a mint endpoint acting on *method_id*.

    Returns ``""`` when permitted, else a short reason. The listing above may
    have hidden the method already, but omission is presentation only — the
    endpoint that actually mints a credential must consult the same chokepoint
    itself (fail-closed: an evaluation error denies). Both halves (capability
    on/off and the ``methods`` ruleset) are checked, mirroring the spawn
    capability's two-step.
    """
    from kiro_crew.platform.governance_profiles import governance_permits

    gate = governance_permits(MOBILE_CONNECT_SCOPE, "", log_warning=False, fail_closed=True)
    if not getattr(gate, "permitted", False):
        return getattr(gate, "reason", "") or "mobile connect disabled by policy"
    scoped = governance_permits(
        MOBILE_CONNECT_SCOPE, f"methods:{method_id}", log_warning=False, fail_closed=True
    )
    if not getattr(scoped, "permitted", False):
        return getattr(scoped, "reason", "") or f"method {method_id!r} disabled by policy"
    return ""
