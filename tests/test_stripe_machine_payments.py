"""Focused tests for the high-level Stripe machine-payments facade."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, cast

import pytest

from mpp import Challenge, Credential, Receipt
from mpp.methods.stripe import create
from mpp.methods.stripe._defaults import MACHINE_PAYMENTS_API_VERSION
from mpp.methods.stripe.machine_payments import _CryptoPaymentRecorder
from mpp.methods.tempo._defaults import (
    CHAIN_ID,
    PATH_USD,
    RPC_URL,
    TESTNET_CHAIN_ID,
    TESTNET_RPC_URL,
    USDC,
)
from mpp.server import Mpp, Validation

TEMPO_ADDRESS = "0x" + "1" * 40


@dataclass
class FakePaymentIntent:
    id: str = "pi_test"
    status: str = "succeeded"


class FakePaymentIntents:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def create(self, body: dict[str, Any], *, options: dict[str, Any]) -> FakePaymentIntent:
        self.calls.append((body, options))
        return FakePaymentIntent()


class AsyncFakePaymentIntents(FakePaymentIntents):
    def create(self, body: dict[str, Any], *, options: dict[str, Any]) -> FakePaymentIntent:
        del body, options
        raise AssertionError("sync create must not be used")

    async def create_async(
        self, body: dict[str, Any], *, options: dict[str, Any]
    ) -> FakePaymentIntent:
        self.calls.append((body, options))
        return FakePaymentIntent()


class FakeStripeClient:
    def __init__(self, payment_intents: FakePaymentIntents | None = None) -> None:
        self.payment_intents = payment_intents or FakePaymentIntents()
        self.v1 = type("V1", (), {"payment_intents": self.payment_intents})()


def machine_payments(**overrides: Any):
    defaults = {
        "network_id": "bn_test",
        "livemode": False,
        "client": FakeStripeClient(),
    }
    defaults.update(overrides)
    return create(**defaults)


def test_default_methods_are_spt_only_without_static_tempo_address() -> None:
    payments = machine_payments()

    methods = payments.default_methods()

    assert [method.name for method in methods] == ["stripe"]
    request = payments.spt.charge().transform_request({"amount": "50"}, None)
    assert request["methodDetails"] == {
        "networkId": "bn_test",
        "paymentMethodTypes": ["card", "link"],
    }


def test_default_methods_advertise_static_tempo_before_spt() -> None:
    payments = machine_payments(deposit_addresses={"tempo": TEMPO_ADDRESS})

    methods = payments.default_methods()

    assert [method.name for method in methods] == ["tempo", "stripe"]
    assert payments.tempo.charge().recipient == TEMPO_ADDRESS


def test_tempo_requires_a_static_address_and_rejects_unknown_networks() -> None:
    with pytest.raises(ValueError, match=r"deposit_addresses\['tempo'\] is required"):
        machine_payments().tempo.charge()
    with pytest.raises(ValueError, match="unsupported deposit address network: base"):
        machine_payments(deposit_addresses={"base": TEMPO_ADDRESS})
    with pytest.raises(ValueError, match="40-hex-character address"):
        machine_payments(deposit_addresses={"tempo": "not-an-address"})


def test_tempo_defaults_are_canonical_and_cannot_be_overridden() -> None:
    testnet = machine_payments(deposit_addresses={"tempo": TEMPO_ADDRESS}).tempo.charge()
    mainnet = machine_payments(
        livemode=True, deposit_addresses={"tempo": TEMPO_ADDRESS}
    ).tempo.charge()

    assert (testnet.chain_id, testnet.currency, testnet.rpc_url) == (
        TESTNET_CHAIN_ID,
        PATH_USD,
        TESTNET_RPC_URL,
    )
    assert (mainnet.chain_id, mainnet.currency, mainnet.rpc_url) == (CHAIN_ID, USDC, RPC_URL)
    assert (testnet.decimals, mainnet.decimals) == (6, 6)
    with pytest.raises(TypeError):
        testnet = machine_payments(deposit_addresses={"tempo": TEMPO_ADDRESS}).tempo.charge(
            chain_id=CHAIN_ID  # type: ignore[call-arg]
        )
        del testnet


def test_methods_enforce_the_fixed_minima() -> None:
    payments = machine_payments(deposit_addresses={"tempo": TEMPO_ADDRESS})
    spt_offer = payments.spt.charge().can_offer
    tempo_offer = payments.tempo.charge().can_offer

    assert spt_offer is not None
    assert tempo_offer is not None
    assert not spt_offer({"amount": "49"})
    assert spt_offer({"amount": "50"})
    assert not tempo_offer({"amount": "9999"})
    assert tempo_offer({"amount": "10000"})


@pytest.mark.asyncio
async def test_spt_uses_injected_client_metadata_and_explicit_allowlist() -> None:
    client = FakeStripeClient()
    method = machine_payments(client=client, metadata={"order": 123}).spt.charge()
    request = method.transform_request({"amount": "100", "currency": "usd"}, None)
    challenge = Challenge.create(
        secret_key="secret",
        realm="api.example.com",
        method="stripe",
        intent="charge",
        request=request,
    )

    await cast(Any, method.intents["charge"]).verify(
        Credential(challenge=challenge.to_echo(), payload={"spt": "spt_test"}), request
    )

    body, options = client.payment_intents.calls[0]
    assert body["shared_payment_granted_token"] == "spt_test"
    assert body["payment_method_types"] == ["card", "link"]
    assert "automatic_payment_methods" not in body
    assert body["metadata"] == {"order": "123"}
    assert not any(key.startswith("mpp_") for key in body["metadata"])
    assert options["stripe_version"] == MACHINE_PAYMENTS_API_VERSION


@pytest.mark.asyncio
async def test_tempo_recorder_rounds_records_metadata_and_uses_tx_hash_idempotency() -> None:
    client = FakeStripeClient()
    recorder = _CryptoPaymentRecorder(client=client, metadata={"order": 123})

    for amount in (4999, 5000, 14999, 15000):
        await recorder(
            {"receipt": Receipt.success(f"0x{amount}"), "request": {"amount": str(amount)}}
        )

    assert [body["amount"] for body, _ in client.payment_intents.calls] == [1, 1, 2]
    body, options = client.payment_intents.calls[-1]
    assert body["metadata"] == {"order": "123"}
    assert body["payment_method_options"]["crypto"] == {
        "mode": "transaction_verification",
        "transaction_verification_options": {
            "network": "tempo",
            "transaction_hash": "0x15000",
        },
    }
    assert options == {
        "idempotency_key": "0x15000",
        "stripe_version": MACHINE_PAYMENTS_API_VERSION,
    }


@pytest.mark.asyncio
async def test_tempo_recording_is_best_effort(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeStripeClient()

    def fail(_body: dict[str, Any], *, options: dict[str, Any]) -> FakePaymentIntent:
        del options
        raise RuntimeError("unavailable")

    client.payment_intents.create = fail  # type: ignore[method-assign]
    await _CryptoPaymentRecorder(client=client)(
        {"receipt": Receipt.success("0xfailure"), "request": {"amount": "10000"}}
    )

    assert "failed to record crypto payment" in caplog.text


class SettledTempoIntent:
    name = "charge"

    async def validate(self, credential: Credential, request: dict[str, Any]) -> Validation:
        return Validation(credential=credential, details={}, intent=self.name, request=request)

    async def broadcast(self, credential: Credential, request: dict[str, Any]) -> Receipt:
        del credential, request
        return Receipt.success("0xtempo-settled", method="tempo")


@pytest.mark.asyncio
async def test_tempo_success_event_records_the_settled_request_via_create_async() -> None:
    payment_intents = AsyncFakePaymentIntents()
    client = FakeStripeClient(payment_intents)
    method = machine_payments(
        client=client, deposit_addresses={"tempo": TEMPO_ADDRESS}
    ).tempo.charge()
    method._intents["charge"] = SettledTempoIntent()
    server = Mpp.create(method=method, realm="api.example.com", secret_key="secret")

    challenge = await server.charge(authorization=None, amount="0.01")
    assert isinstance(challenge, Challenge)
    receipt = await server.broadcast_credential(
        Credential(challenge=challenge.to_echo(), payload={})
    )

    assert receipt.reference == "0xtempo-settled"
    assert payment_intents.calls == [
        (
            {
                "amount": 1,
                "currency": "usd",
                "confirm": True,
                "payment_method_data": {"type": "crypto"},
                "payment_method_types": ["crypto"],
                "payment_method_options": {
                    "crypto": {
                        "mode": "transaction_verification",
                        "transaction_verification_options": {
                            "network": "tempo",
                            "transaction_hash": "0xtempo-settled",
                        },
                    }
                },
            },
            {
                "idempotency_key": "0xtempo-settled",
                "stripe_version": MACHINE_PAYMENTS_API_VERSION,
            },
        )
    ]


def test_facade_requires_an_injected_modern_client() -> None:
    with pytest.raises(TypeError, match="client"):
        create(network_id="bn_test", livemode=False)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="injected StripeClient"):
        create(network_id="bn_test", livemode=False, client=None)  # type: ignore[arg-type]


def test_low_level_metadata_is_keyword_only() -> None:
    from mpp.methods.stripe import stripe as stripe_method

    metadata = inspect.signature(stripe_method).parameters["metadata"]
    assert metadata.kind is inspect.Parameter.KEYWORD_ONLY
