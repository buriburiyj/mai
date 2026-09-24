from dataclasses import dataclass

import keyring
from keyring.errors import KeyringError

from mai.core.providers import Provider

SERVICE_NAME = "mai"


class SecretStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    configured: int
    required: int

    @property
    def complete(self) -> bool:
        return self.configured == self.required

    @property
    def partial(self) -> bool:
        return 0 < self.configured < self.required


class SecretStore:
    def _account_name(self, provider: str, field: str) -> str:
        return f"{provider}:{field}"

    def set(self, provider: str, field: str, value: str) -> None:
        value = value.strip()

        if not value:
            raise SecretStoreError("Credential cannot be empty.")

        try:
            keyring.set_password(
                SERVICE_NAME,
                self._account_name(provider, field),
                value,
            )
        except KeyringError as exc:
            raise SecretStoreError(str(exc)) from exc

    def get(self, provider: str, field: str) -> str | None:
        try:
            return keyring.get_password(
                SERVICE_NAME,
                self._account_name(provider, field),
            )
        except KeyringError as exc:
            raise SecretStoreError(str(exc)) from exc

    def delete(self, provider: str, field: str) -> bool:
        try:
            existing = self.get(provider, field)

            if existing is None:
                return False

            keyring.delete_password(
                SERVICE_NAME,
                self._account_name(provider, field),
            )
            return True
        except KeyringError as exc:
            raise SecretStoreError(str(exc)) from exc

    def status(self, provider: Provider) -> CredentialStatus:
        configured = sum(
            self.get(provider.name, field.name) is not None
            for field in provider.credentials
        )

        return CredentialStatus(
            configured=configured,
            required=len(provider.credentials),
        )
