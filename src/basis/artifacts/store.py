"""Artifact versioning, lineage, approval, and drift detection.

**Built from scratch.** AILedSDLC has `documents`, `chat` and `agent_runs` -
storage, but no versioning, no lineage edges, no approval state and no
retention. ai-core has `documents` with a `supersedes_id`, which is version
chaining but not lineage.

The requirement this implements: artefacts must be ingested, analysed,
generated, stored and baselined by the platform, with re-ingestion when edited
outside it.

Four things that follow from that, and none of which existed:

**Versions are immutable.** An update writes a new row and points it at its
predecessor. You cannot ask "what did we tell the client in March" of a table
that overwrites in place.

**Lineage is a separate edge table, not a parent pointer.** A version chain
answers "what came before this". Lineage answers "what was this *derived
from*", which is a different graph: an FSD derived from three requirements and
a fit/gap has four lineage edges and one version predecessor. Impact analysis -
"the client changed requirement R, what is now stale?" - traverses lineage, not
versions. Arguably the platform's most valuable capability, and it is
impossible without this table.

**Baselining is explicit.** A baseline is a named, frozen set of versions -
what "approved as of the design freeze" means. Without it, "approved" drifts
as individual artefacts get re-approved.

**Drift is detected by content hash.** When an artefact is edited outside the
platform and re-ingested, comparing the incoming hash to the stored one tells
you it changed underneath you. Without the hash there is no way to distinguish
a re-upload of the same file from a substantive external edit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..context import RunContext, require_context
from ..errors import AuthorizationDenied, BasisError
from ..storage import ArtifactRepository

__all__ = [
    "ArtifactConflict",
    "ArtifactState",
    "ArtifactStore",
    "ArtifactVersion",
    "DriftReport",
    "LineageKind",
]


class ArtifactState(str, Enum):
    """Approval lifecycle of one version."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    #: A newer version exists. Set automatically when one is created.
    SUPERSEDED = "superseded"

    @property
    def is_final(self) -> bool:
        return self in (ArtifactState.APPROVED, ArtifactState.REJECTED)


#: Allowed transitions. Encoded rather than checked ad hoc, because an approval
#: state machine that can be driven sideways is not a control.
_TRANSITIONS: Mapping[ArtifactState, frozenset[ArtifactState]] = {
    ArtifactState.DRAFT: frozenset({ArtifactState.IN_REVIEW, ArtifactState.REJECTED}),
    ArtifactState.IN_REVIEW: frozenset(
        {ArtifactState.APPROVED, ArtifactState.REJECTED, ArtifactState.DRAFT}
    ),
    ArtifactState.APPROVED: frozenset({ArtifactState.SUPERSEDED}),
    ArtifactState.REJECTED: frozenset({ArtifactState.DRAFT}),
    ArtifactState.SUPERSEDED: frozenset(),
}


class LineageKind(str, Enum):
    """Why one artefact points at another."""

    #: Generated from it (an FSD from a requirement).
    DERIVED_FROM = "derived_from"
    #: Cites it as evidence (a knowledge chunk).
    CITES = "cites"
    #: A new version of it.
    REVISES = "revises"
    #: Ingested from it (a chunk from an uploaded document).
    EXTRACTED_FROM = "extracted_from"


class ArtifactConflict(BasisError):
    """A write would violate versioning or state rules."""


@dataclass(frozen=True)
class ArtifactVersion:
    """One immutable version of an artefact."""

    id: str
    tenant_id: str
    artifact_key: str
    kind: str
    version: int
    content_hash: str
    state: ArtifactState
    project_id: str | None = None
    title: str | None = None
    content: Any = None
    supersedes_id: str | None = None
    created_by: str | None = None
    approved_by: str | None = None
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftReport:
    """Result of comparing incoming content against the stored version."""

    artifact_key: str
    changed: bool
    stored_version: int | None
    stored_hash: str | None
    incoming_hash: str
    #: True when the stored version was approved - an external edit to an
    #: approved artefact is the case that actually matters, because it means
    #: the baseline no longer reflects reality.
    invalidates_approval: bool = False

    @property
    def needs_reingestion(self) -> bool:
        return self.changed


def content_hash(content: Any) -> str:
    """Stable SHA-256 of an artefact's content.

    ``sort_keys`` matters: without it, two structurally identical dicts hash
    differently depending on insertion order, and every re-ingestion looks like
    an edit. That is the difference between drift detection that works and one
    that cries wolf on every upload.
    """
    if isinstance(content, (bytes, bytearray)):
        payload = bytes(content)
    elif isinstance(content, str):
        payload = content.encode("utf-8")
    else:
        payload = json.dumps(
            content, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ArtifactStore:
    """Versioned artefact storage with lineage and approval.

    Storage lives behind an `ArtifactRepository`, so this class holds only the
    rules: versions are immutable, transitions follow the state machine, an
    identical re-write is a conflict rather than a new version, and a service
    principal may not approve.
    """

    def __init__(
        self,
        repo: ArtifactRepository | None = None,
        *,
        dsn: str | None = None,
    ):
        if repo is None:
            from ..storage.postgres import PgArtifactRepository

            repo = PgArtifactRepository(dsn=dsn)
        self._repo = repo

    # ── writing ────────────────────────────────────────────────────────────

    def create_version(
        self,
        artifact_key: str,
        kind: str,
        content: Any,
        *,
        ctx: RunContext | None = None,
        title: str | None = None,
        derived_from: Sequence[tuple[str, LineageKind]] = (),
        metadata: Mapping[str, Any] | None = None,
        state: ArtifactState = ArtifactState.DRAFT,
    ) -> ArtifactVersion:
        """Write a new version, superseding the current one.

        The repository performs the head-read, supersede, insert and lineage
        writes in one transaction - split across statements, two concurrent
        writers could both create "version 3".
        """
        run = ctx if ctx is not None else require_context()
        digest = content_hash(content)

        result = self._repo.create_version(
            run,
            artifact_key=artifact_key,
            kind=kind,
            content=content,
            content_hash=digest,
            title=title,
            state=state.value,
            metadata=dict(metadata or {}),
            lineage=[(sid, link.value) for sid, link in derived_from],
        )

        if result.get("duplicate"):
            raise ArtifactConflict(
                "artifact %r version %d already has this exact content; "
                "creating an identical version would corrupt the history"
                % (artifact_key, result["version"])
            )

        return ArtifactVersion(
            id=result["id"],
            tenant_id=run.tenant_id,
            artifact_key=artifact_key,
            kind=kind,
            version=result["version"],
            content_hash=digest,
            state=state,
            project_id=run.project_id,
            title=title,
            content=content,
            supersedes_id=result.get("supersedes_id"),
            created_by=run.user_id,
            created_at=result.get("created_at"),
            metadata=dict(metadata or {}),
        )

    def transition(
        self,
        version_id: str,
        to_state: ArtifactState,
        *,
        ctx: RunContext | None = None,
    ) -> ArtifactState:
        """Move a version through the approval state machine.

        Rejects an illegal transition rather than applying it. Approval by a
        service principal is refused for the same reason as a workflow gate -
        the control exists to involve a human.
        """
        run = ctx if ctx is not None else require_context()

        if to_state is ArtifactState.APPROVED and run.principal.is_service:
            raise AuthorizationDenied(
                "artifact",
                "approve",
                "a service principal may not approve an artefact",
            )

        allowed_from = [
            s.value for s, targets in _TRANSITIONS.items() if to_state in targets
        ]
        outcome = self._repo.transition(
            run, version_id, to_state=to_state.value, allowed_from=allowed_from
        )

        if not outcome:
            raise KeyError(
                "no artifact version %s for tenant %s" % (version_id, run.tenant_id)
            )
        if outcome != to_state.value:
            current = ArtifactState(outcome)
            raise ArtifactConflict(
                "cannot move artifact version %s from %s to %s; allowed: %s"
                % (
                    version_id,
                    current.value,
                    to_state.value,
                    ", ".join(sorted(s.value for s in _TRANSITIONS[current]))
                    or "(none - terminal)",
                )
            )
        return to_state

    # ── reading ────────────────────────────────────────────────────────────

    def head(
        self, artifact_key: str, *, ctx: RunContext | None = None
    ) -> dict[str, Any] | None:
        """The current version of an artefact."""
        run = ctx if ctx is not None else require_context()
        return self._repo.head(run, artifact_key)

    def history(
        self, artifact_key: str, *, ctx: RunContext | None = None
    ) -> list[dict[str, Any]]:
        """Every version, newest first."""
        run = ctx if ctx is not None else require_context()
        return self._repo.history(run, artifact_key)

    # ── drift ──────────────────────────────────────────────────────────────

    def check_drift(
        self,
        artifact_key: str,
        incoming_content: Any,
        *,
        ctx: RunContext | None = None,
    ) -> DriftReport:
        """Compare incoming content to the stored head.

        Call this on re-ingestion of anything that could have been edited
        outside the platform. ``invalidates_approval`` is the signal that
        matters: an external edit to an approved artefact means the baseline is
        stale and something downstream needs re-running.
        """
        run = ctx if ctx is not None else require_context()
        digest = content_hash(incoming_content)
        head = self._repo.head(run, artifact_key)

        if head is None:
            return DriftReport(
                artifact_key=artifact_key,
                changed=True,
                stored_version=None,
                stored_hash=None,
                incoming_hash=digest,
            )

        changed = head["content_hash"] != digest
        return DriftReport(
            artifact_key=artifact_key,
            changed=changed,
            stored_version=head["version"],
            stored_hash=head["content_hash"],
            incoming_hash=digest,
            invalidates_approval=changed
            and ArtifactState(head["state"]) is ArtifactState.APPROVED,
        )

    # ── lineage / impact analysis ─────────────────────────────────────────

    def sources_of(
        self, version_id: str, *, ctx: RunContext | None = None
    ) -> list[dict[str, Any]]:
        """What this version was derived from (one hop upstream)."""
        run = ctx if ctx is not None else require_context()
        return self._repo.sources_of(run, version_id)

    def impact_of(
        self,
        version_id: str,
        *,
        ctx: RunContext | None = None,
        max_depth: int = 10,
        exclude_revisions: bool = True,
    ) -> list[dict[str, Any]]:
        """Everything transitively derived from this version.

        The impact-analysis query: "this changed - what is now stale?"

        ``exclude_revisions`` drops REVISES edges by default: a new version of
        an artefact is not "impacted by" its own predecessor in the sense a
        user means when asking what a change broke.
        """
        run = ctx if ctx is not None else require_context()
        kinds = [
            k.value
            for k in LineageKind
            if not (exclude_revisions and k is LineageKind.REVISES)
        ]
        return self._repo.impact_of(
            run, version_id, kinds=kinds, max_depth=max_depth
        )

    # ── baselines ─────────────────────────────────────────────────────────

    def create_baseline(
        self,
        name: str,
        *,
        ctx: RunContext | None = None,
        only_approved: bool = True,
    ) -> int:
        """Freeze the current heads into a named baseline.

        Returns the number of versions captured. ``only_approved`` defaults
        True: a baseline of drafts is not a baseline.
        """
        run = ctx if ctx is not None else require_context()
        states = (
            [ArtifactState.APPROVED.value]
            if only_approved
            else [s.value for s in ArtifactState if s.is_final]
        )
        return self._repo.create_baseline(run, name, states=states)
