"""Coverage for optional payment-method offer and success hooks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import Parameter, signature
from typing import Any, cast

import pytest

from mpp import Challenge, Credential, Receipt
from mpp.events import ServerPaymentSuccessPayload
from mpp.methods.stripe import stripe
from mpp.methods.tempo import tempo
from mpp.server import Mpp, intent
from tests import MockRequest, make_bound_credential

CanOffer = Callable[[dict[str, Any]], bool | Awaitable[bool]]
SuccessHook = Callable[[ServerPaymentSuccessPayload], Any | Awaitable[Any]]


class Method:
    def __init__(
        self,
        name: str,
        *,
        intents: tuple[str, ...] = ("charge",),
        can_offer: CanOffer | None = None,
        on_payment_success: SuccessHook | None = None,
    ) -> None:
        self.name = name
        self.currency = f"{name}-currency"
        self.recipient = f"{name}-recipient"
        self.decimals = 2
        self.can_offer = can_offer
        self.on_payment_success = on_payment_success
        self.intents = {intent_name: self._make_intent(intent_name) for intent_name in intents}

    def _make_intent(self, name: str) -> Any:
        @intent(name=name)
        async def verify(_credential: Credential, _request: dict[str, Any]) -> Receipt:
            return Receipt.success(self.name)

        return verify

    def transform_request(
        self, request: dict[str, Any], _credential: Credential | None
    ) -> dict[str, Any]:
        return {**request, "transformedBy": self.name}

    async def create_credential(self, challenge: Challenge) -> Credential:  # pragma: no cover
        del challenge
        raise NotImplementedError


@pytest.mark.asyncio
async def test_single_method_can_offer_filters_after_transform() -> None:
    requests: list[dict[str, Any]] = []

    async def can_offer(request: dict[str, Any]) -> bool:
        requests.append(request)
        return True

    result = await Mpp.create(
        method=Method("only", can_offer=can_offer),
        realm="api.example.com",
        secret_key="secret",
    ).charge(authorization=None, amount="1.50")

    assert isinstance(result, Challenge)
    assert requests == [
        {
            "amount": "150",
            "currency": "only-currency",
            "recipient": "only-recipient",
            "transformedBy": "only",
        }
    ]


@pytest.mark.asyncio
async def test_multi_method_can_offer_only_advertises_available_methods() -> None:
    seen: list[str] = []

    async def offer_second(request: dict[str, Any]) -> bool:
        seen.append(request["transformedBy"])
        return True

    server = Mpp.create(
        methods=[
            Method(
                "first",
                can_offer=lambda request: seen.append(request["transformedBy"]) or False,
            ),
            Method("second", can_offer=offer_second),
        ],
        realm="api.example.com",
        secret_key="secret",
    )

    result = await server.charge(authorization=None, amount="1.50")

    assert isinstance(result, list)
    assert [challenge.method for challenge in result] == ["second"]
    assert seen == ["first", "second"]


@pytest.mark.asyncio
async def test_pay_checks_can_offer_for_non_payment_auth_and_includes_scope() -> None:
    requests: list[dict[str, Any]] = []
    server = Mpp.create(
        method=Method("only", can_offer=lambda request: requests.append(request) or True),
        realm="api.example.com",
        secret_key="secret",
    )

    @server.pay(amount="1.50")
    async def paid(_request: MockRequest, credential: Credential, receipt: Receipt) -> None:
        del credential, receipt

    response: Any = await paid(MockRequest(authorization="Bearer token", path="/paid"))

    assert response.status_code == 402
    assert requests[0]["transformedBy"] == "only"
    assert requests[0]["_mppx_scope"] == {"resource": "/paid"}


@pytest.mark.asyncio
async def test_payment_credentials_bypass_can_offer() -> None:
    calls = 0

    def can_offer(_request: dict[str, Any]) -> bool:
        nonlocal calls
        calls += 1
        return False

    server = Mpp.create(
        method=Method("only", can_offer=can_offer),
        realm="api.example.com",
        secret_key="secret",
    )
    credential = make_bound_credential(
        payload={},
        request={
            "amount": "150",
            "currency": "only-currency",
            "recipient": "only-recipient",
            "transformedBy": "only",
        },
        secret_key="secret",
        realm="api.example.com",
        method="only",
    )

    result = await server.charge(credential.to_authorization(), "1.50")

    assert isinstance(result, tuple)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [1, "true", None])
async def test_can_offer_requires_a_strict_bool(value: object) -> None:
    server = Mpp.create(
        method=Method("only", can_offer=lambda _request: value),  # type: ignore[arg-type]
        realm="api.example.com",
        secret_key="secret",
    )

    with pytest.raises(ValueError, match="can_offer must return bool"):
        await server.charge(authorization=None, amount="1.50")


@pytest.mark.asyncio
async def test_can_offer_must_be_callable_and_propagates_exceptions() -> None:
    non_callable_server = Mpp.create(
        method=Method("only", can_offer=cast(CanOffer, object())),
        realm="api.example.com",
        secret_key="secret",
    )
    with pytest.raises(ValueError, match="can_offer must be callable"):
        await non_callable_server.charge(authorization=None, amount="1.50")

    def fail_offer(_request: dict[str, Any]) -> bool:
        raise RuntimeError("offer hook failure")

    failing_server = Mpp.create(
        method=Method("only", can_offer=fail_offer),
        realm="api.example.com",
        secret_key="secret",
    )
    with pytest.raises(RuntimeError, match="offer hook failure"):
        await failing_server.charge(authorization=None, amount="1.50")


@pytest.mark.asyncio
async def test_all_unavailable_methods_raise_an_error() -> None:
    server = Mpp.create(
        methods=[
            Method("first", can_offer=lambda _request: False),
            Method("second", can_offer=lambda _request: False),
        ],
        realm="api.example.com",
        secret_key="secret",
    )

    with pytest.raises(ValueError, match="No payment offers"):
        await server.charge(authorization=None, amount="1.50")


def test_method_success_hook_must_be_callable() -> None:
    with pytest.raises(ValueError, match="on_payment_success must be callable"):
        Mpp.create(
            method=Method("only", on_payment_success=cast(SuccessHook, object())),
            realm="api.example.com",
            secret_key="secret",
        )


def _response_challenges(response: Any) -> list[Challenge]:
    if hasattr(response, "raw_headers"):
        values = [
            value.decode()
            for name, value in response.raw_headers
            if name.lower() == b"www-authenticate"
        ]
    else:
        values = response["headers"]["WWW-Authenticate"]
        values = [values] if isinstance(values, str) else values
    return [Challenge.from_www_authenticate(value) for value in values]


@pytest.mark.asyncio
async def test_custom_intent_success_hook_only_runs_for_settled_method() -> None:
    first_calls: list[ServerPaymentSuccessPayload] = []
    second_calls: list[ServerPaymentSuccessPayload] = []
    server = Mpp.create(
        methods=[
            Method("first", intents=("custom",), on_payment_success=first_calls.append),
            Method("second", intents=("custom",), on_payment_success=second_calls.append),
        ],
        realm="api.example.com",
        secret_key="secret",
    )

    @server.pay(amount="1.50", intent="custom")
    async def paid(_request: MockRequest, credential: Credential, receipt: Receipt) -> str:
        assert credential.challenge.method == "second"
        return receipt.reference or ""

    response = await paid(MockRequest(path="/custom"))
    challenges = _response_challenges(response)
    credential = make_bound_credential(
        payload={},
        request=challenges[1].request,
        secret_key="secret",
        realm="api.example.com",
        method="second",
        intent="custom",
    )

    paid_request = MockRequest(authorization=credential.to_authorization(), path="/custom")
    assert await paid(paid_request) == "second"
    assert first_calls == []
    assert second_calls[0]["method"] == "second"
    assert second_calls[0]["intent"] == "custom"


@pytest.mark.asyncio
async def test_global_and_method_success_handlers_receive_same_payload() -> None:
    method_events: list[ServerPaymentSuccessPayload] = []
    global_events: list[ServerPaymentSuccessPayload] = []

    async def record_method_event(payload: ServerPaymentSuccessPayload) -> None:
        method_events.append(payload)

    server = Mpp.create(
        methods=[Method("first"), Method("second", on_payment_success=record_method_event)],
        realm="api.example.com",
        secret_key="secret",
    )
    server.on_payment_success(global_events.append)

    offers = await server.charge(authorization=None, amount="1.50")
    assert isinstance(offers, list)
    credential = make_bound_credential(
        payload={},
        request=offers[1].request,
        secret_key="secret",
        realm="api.example.com",
        method="second",
    )

    result = await server.charge(credential.to_authorization(), "1.50")

    assert isinstance(result, tuple)
    assert method_events == global_events
    assert method_events[0] is global_events[0]


@pytest.mark.asyncio
async def test_success_hook_errors_are_swallowed_and_never_run_for_issuance_or_failure() -> None:
    calls: list[ServerPaymentSuccessPayload] = []

    def fail_after_recording(payload: ServerPaymentSuccessPayload) -> None:
        calls.append(payload)
        raise RuntimeError("hook failure")

    server = Mpp.create(
        method=Method("only", on_payment_success=fail_after_recording),
        realm="api.example.com",
        secret_key="secret",
    )

    challenge = await server.charge(authorization=None, amount="1.50")
    assert isinstance(challenge, Challenge)
    assert calls == []

    failed = await server.charge(authorization="Payment malformed", amount="1.50")
    assert isinstance(failed, Challenge)
    assert calls == []

    credential = make_bound_credential(
        payload={},
        request=challenge.request,
        secret_key="secret",
        realm="api.example.com",
        method="only",
    )
    result = await server.charge(credential.to_authorization(), "1.50")
    assert isinstance(result, tuple)
    assert len(calls) == 1


def test_method_factories_preserve_offer_and_success_hooks() -> None:
    def can_offer(_request: dict[str, Any]) -> bool:
        return True

    def on_payment_success(_payload: ServerPaymentSuccessPayload) -> None:
        pass

    stripe_method = stripe(
        intents={},
        currency="usd",
        recipient="acct_123",
        can_offer=can_offer,
        on_payment_success=on_payment_success,
    )
    tempo_method = tempo(
        intents={},
        currency="0xcurrency",
        recipient="0xrecipient",
        can_offer=can_offer,
        on_payment_success=on_payment_success,
    )

    assert stripe_method.currency == "usd"
    assert stripe_method.recipient == "acct_123"
    assert stripe_method.can_offer is can_offer
    assert stripe_method.on_payment_success is on_payment_success
    assert signature(type(stripe_method)).parameters["can_offer"].kind is Parameter.KEYWORD_ONLY
    assert (
        signature(type(stripe_method)).parameters["on_payment_success"].kind
        is Parameter.KEYWORD_ONLY
    )
    assert tempo_method.currency == "0xcurrency"
    assert tempo_method.recipient == "0xrecipient"
    assert tempo_method.can_offer is can_offer
    assert tempo_method.on_payment_success is on_payment_success
    assert signature(type(tempo_method)).parameters["can_offer"].kind is Parameter.KEYWORD_ONLY
    assert (
        signature(type(tempo_method)).parameters["on_payment_success"].kind
        is Parameter.KEYWORD_ONLY
    )
