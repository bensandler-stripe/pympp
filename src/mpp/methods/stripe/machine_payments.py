"""Small Stripe machine-payments facade for SPT and Tempo."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from mpp.methods.stripe._defaults import MACHINE_PAYMENTS_API_VERSION
from mpp.methods.stripe.client import CanOfferFn, StripeMethod, stripe
from mpp.methods.stripe.intents import ChargeIntent, _resolve_payment_intents
from mpp.methods.tempo._defaults import (
    CHAIN_ID,
    PATH_USD,
    PATH_USD_DECIMALS,
    RPC_URL,
    TESTNET_CHAIN_ID,
    TESTNET_RPC_URL,
    USDC,
)

if TYPE_CHECKING:
    from mpp.methods.tempo.client import TempoMethod
    from mpp.server.method import Method

logger = logging.getLogger(__name__)

SPT_MINIMUM_MINOR_UNITS = 50
RAW_UNITS_PER_CENT = 10_000
_TEMPO_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")


class StripeClient(Protocol):
    """The injected subset of a modern Stripe SDK client used by this facade."""

    @property
    def v1(self) -> object: ...


def _minimum_amount(minimum: int) -> CanOfferFn:
    def can_offer(request: dict[str, Any]) -> bool:
        try:
            return int(request["amount"]) >= minimum
        except (KeyError, TypeError, ValueError):
            return False

    return can_offer


def _metadata(metadata: Mapping[str, object] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping) or not all(isinstance(key, str) for key in metadata):
        raise ValueError("metadata must be a mapping with string keys")
    return {key: str(value) for key, value in metadata.items()}


class _CryptoPaymentRecorder:
    """Best-effort recorder for an already verified Tempo payment."""

    def __init__(
        self,
        *,
        client: StripeClient,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._client = client
        self._metadata = _metadata(metadata)

    async def __call__(self, payload: Mapping[str, Any]) -> None:
        """Record *payload* without allowing Stripe recording to fail settlement."""
        try:
            receipt = payload.get("receipt")
            reference = getattr(receipt, "reference", None)
            if not isinstance(reference, str):
                return

            amount = int(payload["request"]["amount"])
            cents = (amount + (RAW_UNITS_PER_CENT // 2)) // RAW_UNITS_PER_CENT
            if cents < 1:
                return

            body: dict[str, Any] = {
                "amount": cents,
                "currency": "usd",
                "confirm": True,
                "payment_method_data": {"type": "crypto"},
                "payment_method_types": ["crypto"],
                "payment_method_options": {
                    "crypto": {
                        "mode": "transaction_verification",
                        "transaction_verification_options": {
                            "network": "tempo",
                            "transaction_hash": reference,
                        },
                    }
                },
            }
            if self._metadata is not None:
                body["metadata"] = self._metadata

            payment_intents = _resolve_payment_intents(self._client)
            options = {
                "idempotency_key": reference,
                "stripe_version": MACHINE_PAYMENTS_API_VERSION,
            }
            create_async = getattr(payment_intents, "create_async", None)
            if callable(create_async):
                await cast(Any, create_async)(body, options=options)
            else:
                await asyncio.to_thread(payment_intents.create, body, options=options)
        except Exception:
            logger.warning("[stripe] failed to record crypto payment", exc_info=True)


class _SptPayments:
    def __init__(self, parent: MachinePayments) -> None:
        self._parent = parent

    def charge(self) -> StripeMethod:
        return stripe(
            intents={
                "charge": ChargeIntent(
                    client=self._parent._client,
                    include_analytics=False,
                    stripe_version=MACHINE_PAYMENTS_API_VERSION,
                )
            },
            currency="usd",
            recipient=self._parent.network_id,
            network_id=self._parent.network_id,
            payment_method_types=["card", "link"],
            metadata=self._parent.metadata,
            can_offer=_minimum_amount(SPT_MINIMUM_MINOR_UNITS),
        )


class _TempoPayments:
    def __init__(self, parent: MachinePayments) -> None:
        self._parent = parent

    @property
    def configured(self) -> bool:
        return self._parent._tempo_deposit_address is not None

    def charge(self) -> TempoMethod:
        """Create a Tempo charge method using the configured static address."""
        if not self.configured:
            raise ValueError("deposit_addresses['tempo'] is required for Tempo payments")

        from mpp.methods.tempo import ChargeIntent as TempoChargeIntent
        from mpp.methods.tempo import tempo

        chain_id = CHAIN_ID if self._parent.livemode else TESTNET_CHAIN_ID
        currency = USDC if self._parent.livemode else PATH_USD
        rpc_url = RPC_URL if self._parent.livemode else TESTNET_RPC_URL
        return tempo(
            intents={"charge": TempoChargeIntent(chain_id=chain_id, rpc_url=rpc_url)},
            chain_id=chain_id,
            rpc_url=rpc_url,
            currency=currency,
            recipient=self._parent._tempo_deposit_address,
            decimals=PATH_USD_DECIMALS,
            can_offer=_minimum_amount(RAW_UNITS_PER_CENT),
            on_payment_success=_CryptoPaymentRecorder(
                client=self._parent._client,
                metadata=self._parent.metadata,
            ),
        )


class MachinePayments:
    """Configure Stripe SPT and static Tempo machine payments from one client."""

    def __init__(
        self,
        *,
        network_id: str,
        livemode: bool,
        client: StripeClient,
        deposit_addresses: Mapping[str, str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(network_id, str) or not network_id:
            raise ValueError("network_id is required")
        if not isinstance(livemode, bool):
            raise ValueError("livemode must be a bool")
        if client is None or getattr(getattr(client, "v1", None), "payment_intents", None) is None:
            raise ValueError("client must be an injected StripeClient with .v1.payment_intents")
        if deposit_addresses is not None and not isinstance(deposit_addresses, Mapping):
            raise ValueError("deposit_addresses must be a mapping")

        addresses = dict(deposit_addresses or {})
        unsupported = set(addresses).difference({"tempo"})
        if unsupported:
            raise ValueError(f"unsupported deposit address network: {next(iter(unsupported))}")
        tempo_address = addresses.get("tempo")
        if tempo_address is not None and (
            not isinstance(tempo_address, str) or _TEMPO_ADDRESS_RE.fullmatch(tempo_address) is None
        ):
            raise ValueError(
                "deposit_addresses['tempo'] must be a 0x-prefixed 40-hex-character address"
            )

        self.network_id = network_id
        self.livemode = livemode
        self._client = client
        self._tempo_deposit_address = tempo_address
        self.metadata = _metadata(metadata)
        self.spt = _SptPayments(self)
        self.tempo = _TempoPayments(self)

    def default_methods(self) -> list[Method]:
        """Return Tempo (when configured) followed by Stripe SPT."""
        methods: list[Method] = [self.spt.charge()]
        if self.tempo.configured:
            methods.insert(0, self.tempo.charge())
        return methods


def create(
    *,
    network_id: str,
    livemode: bool,
    client: StripeClient,
    deposit_addresses: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MachinePayments:
    """Create the Stripe machine-payments facade from an injected Stripe client."""
    return MachinePayments(
        network_id=network_id,
        livemode=livemode,
        client=client,
        deposit_addresses=deposit_addresses,
        metadata=metadata,
    )
