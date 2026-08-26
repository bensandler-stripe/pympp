"""Focused coverage for multi-method server behavior."""

from __future__ import annotations

from typing import Any

import pytest

from mpp import Challenge, Credential, Receipt
from mpp.errors import InvalidChallengeError, MalformedCredentialError
from mpp.server import Mpp, Validation, intent
from mpp.store import MemoryStore
from tests import MockRequest, make_bound_credential


class Method:
    def __init__(self, name: str) -> None:
        self.name = name
        self.currency = f"{name}-currency"
        self.recipient = f"{name}-recipient"
        self.decimals = 2

        @intent(name="charge")
        async def charge(_credential: Credential, _request: dict[str, Any]) -> Receipt:
            return Receipt.success(name)

        self.intents: dict[str, Any] = {"charge": charge}

    def transform_request(
        self, request: dict[str, Any], _credential: Credential | None
    ) -> dict[str, Any]:
        return {**request, "transformedBy": self.name}

    async def create_credential(self, challenge: Challenge) -> Credential:  # pragma: no cover
        del challenge
        raise NotImplementedError


def test_create_requires_one_distinct_method_argument() -> None:
    first = Method("first")
    second = Method("second")

    server = Mpp.create(methods=[first, second], realm="api.example.com", secret_key="secret")
    assert server.methods == (first, second)
    assert server.method is first

    direct = Mpp(methods=[first, second], realm="api.direct.example", secret_key="secret")
    assert direct.methods == (first, second)

    constructor_positional = Mpp(first, "api.constructor.example", "constructor-secret")
    assert constructor_positional.method is first

    positional = Mpp.create(first, "api.positional.example", "positional-secret")
    assert positional.realm == "api.positional.example"
    assert positional.secret_key == "positional-secret"

    with pytest.raises(ValueError, match="method= or methods="):
        Mpp.create(realm="api.example.com", secret_key="secret")
    with pytest.raises(ValueError, match="not both"):
        Mpp.create(method=first, methods=[second], realm="api.example.com", secret_key="secret")
    with pytest.raises(ValueError, match="duplicate"):
        Mpp.create(methods=[first, first], realm="api.example.com", secret_key="secret")
    with pytest.raises(ValueError, match="not both"):
        Mpp(first, "api.example.com", "secret", methods=[second])
    with pytest.raises(ValueError, match="required"):
        Mpp(None, "api.example.com", "secret", methods=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authorization", "error_type"),
    [
        ("Payment malformed", MalformedCredentialError),
        (
            make_bound_credential(
                payload={},
                request={
                    "amount": "150",
                    "currency": "first-currency",
                    "recipient": "first-recipient",
                },
                secret_key="secret",
                realm="api.example.com",
                method="unknown",
            ).to_authorization(),
            InvalidChallengeError,
        ),
    ],
)
async def test_multi_payment_credentials_keep_typed_failure_events(
    authorization: str, error_type: type[Exception]
) -> None:
    server = Mpp.create(
        methods=[Method("first"), Method("second")],
        realm="api.example.com",
        secret_key="secret",
    )
    failures: list[dict[str, Any]] = []
    server.on_payment_failed(failures.append)

    result = await server.charge(authorization=authorization, amount="1.50")
    assert isinstance(result, Challenge)
    assert isinstance(failures[0]["error"], error_type)


class LifecycleIntent:
    name = "charge"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._store: MemoryStore | None = None

    async def validate(self, credential: Credential, request: dict[str, Any]) -> Validation:
        self.calls.append("validate")
        return Validation(credential=credential, details={}, intent=self.name, request=request)

    async def broadcast(self, _credential: Credential, _request: dict[str, Any]) -> Receipt:
        self.calls.append("broadcast")
        return Receipt.success("second-lifecycle")


@pytest.mark.asyncio
async def test_lifecycle_uses_second_method_and_emits_its_name() -> None:
    second_intent = LifecycleIntent()
    first = Method("first")
    second = Method("second")
    second.intents = {"charge": second_intent}
    server = Mpp.create(methods=[first, second], realm="api.example.com", secret_key="secret")
    events: list[dict[str, Any]] = []
    server.on_payment_success(events.append)
    credential = make_bound_credential(
        payload={},
        request={},
        secret_key="secret",
        realm="api.example.com",
        method="second",
    )

    await server.validate_credential(credential)
    receipt = await server.broadcast_credential(credential)
    assert receipt.reference == "second-lifecycle"
    assert second_intent.calls == ["validate", "validate", "broadcast"]
    assert events[0]["method"] == "second"


def test_multi_method_store_wiring() -> None:
    first_intent = LifecycleIntent()
    second_intent = LifecycleIntent()
    first = Method("first")
    second = Method("second")
    first.intents = {"charge": first_intent}
    second.intents = {"charge": second_intent}
    store = MemoryStore()

    Mpp.create(
        methods=[first, second],
        realm="api.example.com",
        secret_key="secret",
        store=store,
    )

    assert first_intent._store is store
    assert second_intent._store is store


@pytest.mark.asyncio
async def test_multi_charge_orders_offers_and_redeems_echoed_method() -> None:
    events: list[dict[str, Any]] = []
    first = Method("first")
    second = Method("second")
    server = Mpp.create(
        methods=[first, second],
        realm="api.example.com",
        secret_key="secret",
    )
    server.on_payment_success(events.append)

    result = await server.charge(authorization=None, amount="1.50")
    assert isinstance(result, list)
    assert [challenge.method for challenge in result] == ["first", "second"]

    credential = make_bound_credential(
        payload={},
        request=result[1].request,
        secret_key="secret",
        realm="api.example.com",
        method="second",
    )
    payment = await server.charge(authorization=credential.to_authorization(), amount="1.50")
    assert isinstance(payment, tuple)
    assert payment[1].reference == "second"
    assert events[0]["method"] == "second"


@pytest.mark.asyncio
async def test_multi_pay_appends_challenges_and_routes_to_echoed_method() -> None:
    first = Method("first")
    second = Method("second")
    server = Mpp.create(
        methods=[first, second],
        realm="api.example.com",
        secret_key="secret",
    )

    @server.pay(amount="1.50")
    async def paid(_request: MockRequest, credential: Credential, receipt: Receipt) -> str:
        assert credential.challenge.method == "second"
        return receipt.reference or ""

    response: Any = await paid(MockRequest(path="/paid"))
    if hasattr(response, "raw_headers"):
        values = [
            value.decode()
            for name, value in response.raw_headers
            if name.lower() == b"www-authenticate"
        ]
    else:
        values = response["headers"]["WWW-Authenticate"]
        values = [values] if isinstance(values, str) else values
    offered_methods = [Challenge.from_www_authenticate(value).method for value in values]
    assert offered_methods == ["first", "second"]

    challenge = Challenge.from_www_authenticate(values[1])
    credential = make_bound_credential(
        payload={},
        request=challenge.request,
        secret_key="secret",
        realm="api.example.com",
        method="second",
    )
    paid_request = MockRequest(authorization=credential.to_authorization(), path="/paid")
    assert await paid(paid_request) == "second"
