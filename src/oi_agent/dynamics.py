"""Typed behavioral memory models, strict validators, and compact serialization."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MAX_PROFILE_JSON_BYTES = 4096
MAX_TRANSIENT_JSON_BYTES = 2048
MAX_PULSE_JSON_BYTES = 2048
MAX_CALIBRATION_JSON_BYTES = 2048

DetailPreference = Literal["brief", "balanced", "detailed"]
ChallengePreference = Literal["gentle", "direct", "adversarial"]
HumorPreference = Literal["low", "medium", "high"]
DecisionStyle = Literal["options", "recommendation", "execution_first"]
EnergyBand = Literal["low", "steady", "high", "frustrated", "unknown"]
MomentumBand = Literal[
    "blocked", "planning", "building", "debugging", "shipping", "celebrating", "unknown"
]
VerbosityBand = Literal["concise", "balanced", "detailed"]
DirectnessBand = Literal["direct", "standard", "gentle"]

DETAIL_PREFERENCES = {"brief", "balanced", "detailed"}
CHALLENGE_PREFERENCES = {"gentle", "direct", "adversarial"}
HUMOR_PREFERENCES = {"low", "medium", "high"}
DECISION_STYLES = {"options", "recommendation", "execution_first"}
ENERGY_BANDS = {"low", "steady", "high", "frustrated", "unknown"}
MOMENTUM_BANDS = {
    "blocked", "planning", "building", "debugging", "shipping", "celebrating", "unknown"
}
VERBOSITY_BANDS = {"concise", "balanced", "detailed"}
DIRECTNESS_BANDS = {"direct", "standard", "gentle"}

ALLOWLISTED_FRICTIONS = {
    "ci_cd",
    "architecture",
    "dependencies",
    "external_api",
    "scope_creep",
    "unclear_spec",
    "technical_debt",
    "communication",
    "testing",
    "none",
}

ALLOWLISTED_FORMATTING_AVOIDANCES = {
    "corporate_filler",
    "unsolicited_summaries",
    "forced_bullet_lists",
    "hedging",
    "cheerleading",
    "apologies",
    "tldr",
}

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MARKUP_CHAR_RE = re.compile(r"[<>]")

# Free-label policy (GAP 2): short, normalized, bounded-charset labels only.
TOPIC_LABEL_MAX_CHARS = 32
TOPIC_LABEL_MAX_WORDS = 3
TOPIC_LABEL_MAX_WORD_CHARS = 24

_TOPIC_CHARSET_RE = re.compile(r"[a-z0-9 _-]+\Z")
# Secret/excerpt markers: credential-shaped words plus URL/email punctuation.
_TOPIC_SECRET_MARKERS: tuple[str, ...] = (
    "://",
    "@",
    "password",
    "passwd",
    "pwd",
    "secret",
    "credential",
    "token",
    "apikey",
    "api_key",
    "bearer",
    "oauth",
    "ssh",
    "private key",
    "access key",
    "auth",
    "otp",
    "2fa",
)
_TOPIC_URL_RE = re.compile(r"\S+://\S*|www\.|\S+\.\S+/\S*")
_TOPIC_EMAIL_RE = re.compile(r"\S+@\S+")

# Confidence at or below this floor is considered unrenderable/unknown (PIN-1/PIN-2).
CONFIDENCE_FLOOR: float = 0.35

# --- Deterministic merge constants (audit F2/F3) ---------------------------------
# Stable profile fields decay with a 30-day half-life; observations older than
# that contribute proportionally less when merging.
PROFILE_RECENCY_HALF_LIFE_SECONDS = 30 * 86400
# Team pulse records hard-expire after PULSE_TTL_SECONDS (4h); within that window
# stale evidence is downweighted with a half-life of half the TTL.
PULSE_RECENCY_HALF_LIFE_SECONDS = 2 * 3600
# Calibration records hard-expire after CALIBRATION_TTL_SECONDS (24h); a 12h
# half-life downweights stale evidence inside the window.
CALIBRATION_RECENCY_HALF_LIFE_SECONDS = 12 * 3600
# Per-observation floor on the recency factor so one merge can never collapse
# stored confidence in a single step (bounded per-observation movement).
MIN_RECENCY_FACTOR = 0.1

# Maximum movement of the numeric directness score per observation.
MAX_NUMERIC_SHIFT = 0.15
# Smoothing constant so a single observation can never fully dominate the weight.
NUMERIC_WEIGHT_SMOOTHING = 0.1
# A categorical value never flips on a single contradiction (GAP 4): a
# contradiction only reduces the current field's confidence by a bounded step
# scaled by observation confidence. A flip is allowed only once the current
# field's effective (recency-decayed) confidence has already decayed below
# CATEGORICAL_FLIP_CONF_FLOOR AND the contradicting observation confidence
# exceeds CATEGORICAL_FLIP_MIN_CONFIDENCE.
CATEGORICAL_FLIP_CONF_FLOOR = 0.35
CATEGORICAL_FLIP_MIN_CONFIDENCE = 0.8
CONSISTENT_CONF_GAIN = 0.1
CONTRADICTION_CONF_PENALTY = 0.15
CONTRADICTION_CONF_FLOOR = 0.05
FLIP_CONFIDENCE_FACTOR = 0.6

# Exact memory document schema version enforced by every validator (GAP 6).
MEMORY_SCHEMA_VERSION = 2

MAX_RECURRING_TOPICS = 5
TRANSIENT_TTL_SECONDS = 4 * 3600
PULSE_TTL_SECONDS = 4 * 3600
CALIBRATION_TTL_SECONDS = 24 * 3600


def confidence_band(value: float) -> str:
    """Return "low" (<0.5), "medium" (<0.8) or "high" (>=0.8) for a confidence value."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or math.isnan(float(value))
        or math.isinf(float(value))
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError("confidence_band requires a finite value in [0, 1]")
    if value < 0.5:
        return "low"
    if value < 0.8:
        return "medium"
    return "high"


def directness_band(value: float) -> str:
    """Normalize a numeric directness score to a render-safe band string."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or math.isnan(float(value))
        or math.isinf(float(value))
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError("directness_band requires a finite value in [0, 1]")
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def recency_factor(now: int, last_ts: int, half_life_seconds: int) -> float:
    """Deterministic exponential recency factor in (MIN_RECENCY_FACTOR, 1.0].

    A factor of 1.0 means the prior evidence is fresh; values decay toward
    MIN_RECENCY_FACTOR with the given half-life. Future timestamps are treated
    as maximally fresh.
    """
    if half_life_seconds <= 0:
        return 1.0
    age = max(0, int(now) - int(last_ts))
    return max(MIN_RECENCY_FACTOR, 0.5 ** (age / half_life_seconds))


def sanitize_label(text: Any, max_len: int = 40) -> str:
    """Validate and sanitize a short text label, rejecting markup and control characters.

    Error messages are bounded and never echo the rejected content.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string label, got {type(text).__name__}")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty label not allowed")
    if len(cleaned) > max_len:
        raise ValueError(f"Label exceeds maximum length of {max_len}")
    if _CONTROL_CHAR_RE.search(cleaned):
        raise ValueError("Control characters detected in label")
    if _MARKUP_CHAR_RE.search(cleaned):
        raise ValueError("Markup characters (<, >) detected in label")
    return cleaned


def _looks_like_secret_or_excerpt(normalized: str) -> bool:
    """Heuristic rejection of credential-shaped strings, URLs, and excerpts."""
    if any(marker in normalized for marker in _TOPIC_SECRET_MARKERS):
        return True
    if _TOPIC_URL_RE.search(normalized) or _TOPIC_EMAIL_RE.search(normalized):
        return True
    for token in normalized.split():
        if (
            len(token) >= 20
            and any(c.isalpha() for c in token)
            and any(c.isdigit() for c in token)
        ):
            return True
    return False


def sanitize_topic_label(text: Any) -> str:
    """Normalize and validate a free-form topic/focus label (GAP 2).

    Applied to member observation ``topics``/``current_focus`` and team pulse
    ``focus_areas`` at BOTH observation-validation and profile-validation time
    so neither write path can persist raw text.

    This is a bounded-charset + secret-scanning mitigation, not a semantic
    guarantee: its purpose is that verbatim sentences, credential-shaped
    strings, URLs, and quoted excerpts cannot persist as "labels". Labels are
    lowercased with internal whitespace collapsed, restricted to
    ``[a-z0-9 _-]``, bounded to 32 chars / 1-3 words / 24 chars per word, and
    rejected when they match secret/excerpt heuristics. Error messages are
    bounded and never echo the rejected content. Non-string input is rejected
    through the same bounded empty-input path (single rejection class).
    """
    normalized = " ".join(text.split()).lower() if isinstance(text, str) else ""
    if not normalized:
        raise ValueError("Empty topic label not allowed")
    if len(normalized) > TOPIC_LABEL_MAX_CHARS:
        raise ValueError(
            f"Topic label exceeds maximum length of {TOPIC_LABEL_MAX_CHARS}"
        )
    words = normalized.split()
    if not 1 <= len(words) <= TOPIC_LABEL_MAX_WORDS:
        raise ValueError(
            f"Topic label must contain 1 to {TOPIC_LABEL_MAX_WORDS} words"
        )
    if any(len(word) > TOPIC_LABEL_MAX_WORD_CHARS for word in words):
        raise ValueError(
            f"Topic label words must be at most {TOPIC_LABEL_MAX_WORD_CHARS} characters"
        )
    if not _TOPIC_CHARSET_RE.match(normalized):
        raise ValueError("Topic label contains disallowed characters")
    if _looks_like_secret_or_excerpt(normalized):
        raise ValueError("Topic label matches secret/excerpt heuristics")
    return normalized


def normalize_valid_topic_label(text: Any) -> str | None:
    """Return the normalized label when it passes the topic policy, else None."""
    try:
        return sanitize_topic_label(text)
    except ValueError:
        return None


def _validate_float_bounds(val: Any, name: str, low: float = 0.0, high: float = 1.0) -> float:
    """Ensure numeric value is a finite float within bounds."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a float, got {type(val).__name__}")
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"{name} must be finite")
    if fval < low or fval > high:
        raise ValueError(f"{name} must be between {low} and {high}, got {fval}")
    return round(fval, 3)


def _validate_int(val: Any, name: str, min_val: int = 0) -> int:
    """Ensure integer value is non-negative and finite."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an int, got {type(val).__name__}")
    if val < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {val}")
    return val


def _validate_confidence_positive(val: Any, name: str) -> float:
    """Ensure a finite confidence strictly inside (0, 1] (audit F4)."""
    conf = _validate_float_bounds(val, name)
    if conf <= 0.0:
        raise ValueError(f"{name} must be within (0, 1]")
    return conf


@dataclass(frozen=True)
class CommunicationProfile:
    """Stable communication preference parameters.

    Unobserved dimensions are None (GAP 3: never fabricated from defaults);
    their companion confidence is 0.0 and evidence_count 0, so the
    CONFIDENCE_FLOOR rendering path treats them as absent.
    """

    directness: float | None
    detail_preference: DetailPreference | None
    challenge_preference: ChallengePreference | None
    humor_preference: HumorPreference | None
    decision_style: DecisionStyle | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COMMUNICATION_KEYS = {
    "directness",
    "detail_preference",
    "challenge_preference",
    "humor_preference",
    "decision_style",
}


@dataclass(frozen=True)
class MemberProfile:
    """Scope-isolated, versioned stable behavioral profile for a team member."""

    schema_version: int
    communication: CommunicationProfile
    confidence: dict[str, float]
    evidence_count: dict[str, int]
    last_observed_at: dict[str, int]
    recurring_topics: list[str]
    created_at: int
    updated_at: int
    last_interaction_at: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["communication"] = self.communication.to_dict()
        return d


@dataclass(frozen=True)
class TransientMemberState:
    """Short-lived, expiring state for a team member (e.g. energy band, current focus)."""

    energy: EnergyBand
    current_focus: list[str]
    observed_at: int
    expires_at: int
    schema_version: int = MEMORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamPulse:
    """Short-lived, expiring momentum and friction metrics for a team scope."""

    schema_version: int
    focus_areas: list[str]
    friction_categories: list[str]
    momentum: MomentumBand
    confidence: float
    evidence_count: int
    observed_at: int
    expires_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentCalibration:
    """Short-lived, expiring response calibration directives for a team scope."""

    schema_version: int
    preferred_verbosity: VerbosityBand
    preferred_directness: DirectnessBand
    formatting_avoidances: list[str]
    confidence: float
    evidence_count: int
    updated_at: int
    expires_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_compact_json(obj: Any) -> str:
    """Serialize object or dataclass to canonical compact JSON without extraneous spaces."""
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError(f"Cannot serialize object of type {type(obj).__name__}")
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def validate_communication_profile(data: dict[str, Any]) -> CommunicationProfile:
    """Validate communication profile dictionary strictly.

    Unobserved dimensions may be None (GAP 3); any present value must match
    the strict bounds/enum allowlist.
    """
    if not isinstance(data, dict):
        raise ValueError("communication must be a dict")
    unknown = set(data.keys()) - COMMUNICATION_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in communication: {sorted(unknown)}")

    raw_directness = data.get("directness")
    directness = (
        None
        if raw_directness is None
        else _validate_float_bounds(raw_directness, "directness")
    )

    detail = data.get("detail_preference")
    if detail is not None and detail not in DETAIL_PREFERENCES:
        raise ValueError("Invalid detail_preference in profile")

    challenge = data.get("challenge_preference")
    if challenge is not None and challenge not in CHALLENGE_PREFERENCES:
        raise ValueError("Invalid challenge_preference in profile")

    humor = data.get("humor_preference")
    if humor is not None and humor not in HUMOR_PREFERENCES:
        raise ValueError("Invalid humor_preference in profile")

    decision = data.get("decision_style")
    if decision is not None and decision not in DECISION_STYLES:
        raise ValueError("Invalid decision_style in profile")

    return CommunicationProfile(
        directness=directness,
        detail_preference=detail,  # type: ignore[arg-type]
        challenge_preference=challenge,  # type: ignore[arg-type]
        humor_preference=humor,  # type: ignore[arg-type]
        decision_style=decision,  # type: ignore[arg-type]
    )


def _require_schema_version(data: dict[str, Any]) -> int:
    """Enforce exactly the supported memory schema version (GAP 6)."""
    ver = _validate_int(
        data.get("schema_version", MEMORY_SCHEMA_VERSION), "schema_version", 0
    )
    if ver != MEMORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported memory schema_version: {ver} "
            f"(supported: {MEMORY_SCHEMA_VERSION})"
        )
    return ver


def validate_member_profile(data: dict[str, Any]) -> MemberProfile:
    """Validate full member profile document strictly against allowlist schema."""
    if not isinstance(data, dict):
        raise ValueError("MemberProfile data must be a dict")
    allowed_keys = {
        "schema_version",
        "communication",
        "confidence",
        "evidence_count",
        "last_observed_at",
        "recurring_topics",
        "created_at",
        "updated_at",
        "last_interaction_at",
    }
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown keys in MemberProfile: {sorted(unknown)}")

    schema_ver = _require_schema_version(data)
    comm = validate_communication_profile(data.get("communication", {}))

    # Validate confidence map
    raw_conf = data.get("confidence", {})
    if not isinstance(raw_conf, dict):
        raise ValueError("confidence must be a dict")
    if set(raw_conf.keys()) - COMMUNICATION_KEYS:
        raise ValueError(f"Unknown confidence keys: {sorted(set(raw_conf.keys()) - COMMUNICATION_KEYS)}")
    conf = {k: _validate_float_bounds(raw_conf[k], f"confidence.{k}") for k in raw_conf}

    # Validate evidence_count map (zero evidence marks an unobserved dimension,
    # GAP 3: stored alongside confidence 0.0 and no value).
    raw_ev = data.get("evidence_count", {})
    if not isinstance(raw_ev, dict):
        raise ValueError("evidence_count must be a dict")
    if set(raw_ev.keys()) - COMMUNICATION_KEYS:
        raise ValueError(f"Unknown evidence_count keys: {sorted(set(raw_ev.keys()) - COMMUNICATION_KEYS)}")
    ev = {k: _validate_int(raw_ev[k], f"evidence_count.{k}", min_val=0) for k in raw_ev}

    # Validate last_observed_at map
    raw_obs = data.get("last_observed_at", {})
    if not isinstance(raw_obs, dict):
        raise ValueError("last_observed_at must be a dict")
    if set(raw_obs.keys()) - COMMUNICATION_KEYS:
        raise ValueError(f"Unknown last_observed_at keys: {sorted(set(raw_obs.keys()) - COMMUNICATION_KEYS)}")
    obs = {k: _validate_int(raw_obs[k], f"last_observed_at.{k}", min_val=0) for k in raw_obs}
    if set(obs.keys()) - conf.keys():
        raise ValueError(
            f"last_observed_at keys exceed confidence keys: {sorted(set(obs.keys()) - conf.keys())}"
        )

    # Cross-check (audit F13): every dimension present in confidence must have a
    # matching evidence entry and vice versa, same strictness as the typed path.
    if set(conf.keys()) != set(ev.keys()):
        raise ValueError("confidence and evidence_count must cover exactly the same dimensions")

    # Validate recurring_topics (GAP 2 topic-label policy, profile write path)
    raw_topics = data.get("recurring_topics", [])
    if not isinstance(raw_topics, list):
        raise ValueError("recurring_topics must be a list")
    if len(raw_topics) > MAX_RECURRING_TOPICS:
        raise ValueError(f"recurring_topics exceeds limit of {MAX_RECURRING_TOPICS} items (got {len(raw_topics)})")
    topics = [sanitize_topic_label(t) for t in raw_topics]

    created_at = _validate_int(data.get("created_at"), "created_at", min_val=0)
    updated_at = _validate_int(data.get("updated_at"), "updated_at", min_val=0)
    last_interaction = _validate_int(data.get("last_interaction_at"), "last_interaction_at", min_val=0)

    profile = MemberProfile(
        schema_version=schema_ver,
        communication=comm,
        confidence=conf,
        evidence_count=ev,
        last_observed_at=obs,
        recurring_topics=topics,
        created_at=created_at,
        updated_at=updated_at,
        last_interaction_at=last_interaction,
    )
    compact = to_compact_json(profile)
    if len(compact.encode("utf-8")) > MAX_PROFILE_JSON_BYTES:
        raise ValueError(f"MemberProfile serialized size exceeds {MAX_PROFILE_JSON_BYTES} bytes")
    return profile


def validate_transient_member_state(data: dict[str, Any]) -> TransientMemberState:
    """Validate transient member state strictly."""
    if not isinstance(data, dict):
        raise ValueError("TransientMemberState must be a dict")
    allowed_keys = {
        "schema_version",
        "energy",
        "current_focus",
        "observed_at",
        "expires_at",
    }
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown keys in TransientMemberState: {sorted(unknown)}")

    schema_ver = _require_schema_version(data)

    energy = data.get("energy")
    if energy not in ENERGY_BANDS:
        raise ValueError("Invalid energy band in state")

    raw_focus = data.get("current_focus", [])
    if not isinstance(raw_focus, list):
        raise ValueError("current_focus must be a list")
    if len(raw_focus) > 3:
        raise ValueError(f"current_focus exceeds limit of 3 items (got {len(raw_focus)})")
    focus = [sanitize_topic_label(f) for f in raw_focus]

    observed_at = _validate_int(data.get("observed_at"), "observed_at", min_val=0)
    expires_at = _validate_int(data.get("expires_at"), "expires_at", min_val=0)

    state = TransientMemberState(
        energy=energy,  # type: ignore[arg-type]
        current_focus=focus,
        observed_at=observed_at,
        expires_at=expires_at,
        schema_version=schema_ver,
    )
    compact = to_compact_json(state)
    if len(compact.encode("utf-8")) > MAX_TRANSIENT_JSON_BYTES:
        raise ValueError(f"TransientMemberState serialized size exceeds {MAX_TRANSIENT_JSON_BYTES} bytes")
    return state


def validate_team_pulse(data: dict[str, Any]) -> TeamPulse:
    """Validate team pulse strictly."""
    if not isinstance(data, dict):
        raise ValueError("TeamPulse must be a dict")
    allowed_keys = {
        "schema_version",
        "focus_areas",
        "friction_categories",
        "momentum",
        "confidence",
        "evidence_count",
        "observed_at",
        "expires_at",
    }
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown keys in TeamPulse: {sorted(unknown)}")

    schema_ver = _require_schema_version(data)

    raw_focus = data.get("focus_areas", [])
    if not isinstance(raw_focus, list):
        raise ValueError("focus_areas must be a list")
    if len(raw_focus) > 5:
        raise ValueError(f"focus_areas exceeds limit of 5 items (got {len(raw_focus)})")
    focus = [sanitize_topic_label(f) for f in raw_focus]

    raw_frictions = data.get("friction_categories", [])
    if not isinstance(raw_frictions, list):
        raise ValueError("friction_categories must be a list")
    for fc in raw_frictions:
        if fc not in ALLOWLISTED_FRICTIONS:
            raise ValueError("Invalid friction category in pulse")

    momentum = data.get("momentum")
    if momentum not in MOMENTUM_BANDS:
        raise ValueError("Invalid momentum in pulse")

    confidence = _validate_float_bounds(data.get("confidence"), "confidence")
    evidence_count = _validate_int(data.get("evidence_count"), "evidence_count", min_val=1)
    observed_at = _validate_int(data.get("observed_at"), "observed_at", min_val=0)
    expires_at = _validate_int(data.get("expires_at"), "expires_at", min_val=0)

    pulse = TeamPulse(
        schema_version=schema_ver,
        focus_areas=focus,
        friction_categories=list(raw_frictions),
        momentum=momentum,  # type: ignore[arg-type]
        confidence=confidence,
        evidence_count=evidence_count,
        observed_at=observed_at,
        expires_at=expires_at,
    )
    compact = to_compact_json(pulse)
    if len(compact.encode("utf-8")) > MAX_PULSE_JSON_BYTES:
        raise ValueError(f"TeamPulse serialized size exceeds {MAX_PULSE_JSON_BYTES} bytes")
    return pulse


def validate_agent_calibration(data: dict[str, Any]) -> AgentCalibration:
    """Validate agent calibration strictly."""
    if not isinstance(data, dict):
        raise ValueError("AgentCalibration must be a dict")
    allowed_keys = {
        "schema_version",
        "preferred_verbosity",
        "preferred_directness",
        "formatting_avoidances",
        "confidence",
        "evidence_count",
        "updated_at",
        "expires_at",
    }
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown keys in AgentCalibration: {sorted(unknown)}")

    schema_ver = _require_schema_version(data)

    verbosity = data.get("preferred_verbosity")
    if verbosity not in VERBOSITY_BANDS:
        raise ValueError("Invalid preferred_verbosity in calibration")

    directness = data.get("preferred_directness")
    if directness not in DIRECTNESS_BANDS:
        raise ValueError("Invalid preferred_directness in calibration")

    raw_avoidances = data.get("formatting_avoidances", [])
    if not isinstance(raw_avoidances, list):
        raise ValueError("formatting_avoidances must be a list")
    for fa in raw_avoidances:
        if fa not in ALLOWLISTED_FORMATTING_AVOIDANCES:
            raise ValueError("Invalid formatting avoidance in calibration")

    confidence = _validate_float_bounds(data.get("confidence"), "confidence")
    evidence_count = _validate_int(data.get("evidence_count"), "evidence_count", min_val=1)
    updated_at = _validate_int(data.get("updated_at"), "updated_at", min_val=0)
    expires_at = _validate_int(data.get("expires_at"), "expires_at", min_val=0)

    calibration = AgentCalibration(
        schema_version=schema_ver,
        preferred_verbosity=verbosity,  # type: ignore[arg-type]
        preferred_directness=directness,  # type: ignore[arg-type]
        formatting_avoidances=list(raw_avoidances),
        confidence=confidence,
        evidence_count=evidence_count,
        updated_at=updated_at,
        expires_at=expires_at,
    )
    compact = to_compact_json(calibration)
    if len(compact.encode("utf-8")) > MAX_CALIBRATION_JSON_BYTES:
        raise ValueError(f"AgentCalibration serialized size exceeds {MAX_CALIBRATION_JSON_BYTES} bytes")
    return calibration


@dataclass(frozen=True)
class MemberObservation:
    """Typed observation delta for a member from a single interaction or bootstrap."""

    directness: float | None = None
    detail_preference: DetailPreference | None = None
    challenge_preference: ChallengePreference | None = None
    humor_preference: HumorPreference | None = None
    decision_style: DecisionStyle | None = None
    confidence: float = 0.5
    topics: list[str] = field(default_factory=list)
    energy: EnergyBand | None = None
    current_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamPulseObservation:
    """Typed observation delta for team pulse."""

    focus_areas: list[str] = field(default_factory=list)
    friction_categories: list[str] = field(default_factory=list)
    momentum: MomentumBand = "unknown"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentCalibrationObservation:
    """Typed observation delta for agent calibration."""

    preferred_verbosity: VerbosityBand = "concise"
    preferred_directness: DirectnessBand = "direct"
    formatting_avoidances: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MEMBER_STABLE_FIELD_KEYS = (
    "directness",
    "detail_preference",
    "challenge_preference",
    "humor_preference",
    "decision_style",
    "topics",
)
_MEMBER_TRANSIENT_FIELD_KEYS = ("energy", "current_focus")


def _present(value: Any) -> bool:
    """Return whether an optional observation value carries usable content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(isinstance(item, str) and item.strip() for item in value)
    return True


def has_member_observation_stable_fields(data: Any) -> bool:
    """Return whether the observation carries at least one stable-profile field."""
    if not isinstance(data, dict):
        return False
    return any(_present(data.get(key)) for key in _MEMBER_STABLE_FIELD_KEYS)


def has_member_observation_fields(data: Any) -> bool:
    """Return whether a raw member observation carries any observed field at all."""
    if not isinstance(data, dict):
        return False
    return has_member_observation_stable_fields(data) or any(
        _present(data.get(key)) for key in _MEMBER_TRANSIENT_FIELD_KEYS
    )


def has_team_pulse_observation_fields(data: Any) -> bool:
    """Return whether a raw team pulse observation carries usable content."""
    if not isinstance(data, dict):
        return False
    if _present(data.get("focus_areas")) or _present(data.get("friction_categories")):
        return True
    momentum = data.get("momentum")
    return isinstance(momentum, str) and momentum.strip() not in ("", "unknown")


def has_agent_calibration_observation_fields(data: Any) -> bool:
    """Return whether a raw calibration observation carries at least one directive."""
    if not isinstance(data, dict):
        return False
    keys = ("preferred_verbosity", "preferred_directness", "formatting_avoidances")
    return any(_present(data.get(key)) for key in keys)


def validate_member_observation(data: dict[str, Any]) -> MemberObservation:
    """Strictly validate an observation dictionary for a member.

    Observations carrying zero observed fields are rejected (audit F4) and the
    shared observation confidence must lie strictly inside (0, 1].
    """
    if not isinstance(data, dict):
        raise ValueError("MemberObservation must be a dict")
    allowed = {
        "member_id",
        "handle",
        "directness",
        "detail_preference",
        "challenge_preference",
        "humor_preference",
        "decision_style",
        "confidence",
        "topics",
        "energy",
        "current_focus",
    }
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in MemberObservation: {sorted(unknown)}")

    directness = None
    if "directness" in data and data["directness"] is not None:
        directness = _validate_float_bounds(data["directness"], "directness")

    detail = data.get("detail_preference")
    if detail is not None and detail not in DETAIL_PREFERENCES:
        raise ValueError("Invalid detail_preference in observation")

    challenge = data.get("challenge_preference")
    if challenge is not None and challenge not in CHALLENGE_PREFERENCES:
        raise ValueError("Invalid challenge_preference in observation")

    humor = data.get("humor_preference")
    if humor is not None and humor not in HUMOR_PREFERENCES:
        raise ValueError("Invalid humor_preference in observation")

    decision = data.get("decision_style")
    if decision is not None and decision not in DECISION_STYLES:
        raise ValueError("Invalid decision_style in observation")

    raw_conf = data.get("confidence", 0.5)
    conf = _validate_confidence_positive(raw_conf, "confidence")

    raw_topics = data.get("topics", [])
    if not isinstance(raw_topics, list):
        raise ValueError("topics must be a list")
    if len(raw_topics) > MAX_RECURRING_TOPICS:
        raise ValueError(
            f"topics exceeds limit of {MAX_RECURRING_TOPICS} items (got {len(raw_topics)})"
        )
    topics = [sanitize_topic_label(t) for t in raw_topics]

    energy = data.get("energy")
    if energy is not None and energy not in ENERGY_BANDS:
        raise ValueError("Invalid energy band in observation")

    raw_focus = data.get("current_focus", [])
    if not isinstance(raw_focus, list):
        raise ValueError("current_focus must be a list")
    if len(raw_focus) > 3:
        raise ValueError(f"current_focus exceeds limit of 3 items (got {len(raw_focus)})")
    focus = [sanitize_topic_label(f) for f in raw_focus]

    if not has_member_observation_fields(data):
        raise ValueError("MemberObservation carries no observed fields")

    return MemberObservation(
        directness=directness,
        detail_preference=detail,
        challenge_preference=challenge,
        humor_preference=humor,
        decision_style=decision,
        confidence=conf,
        topics=topics,
        energy=energy,
        current_focus=focus,
    )


def validate_team_pulse_observation(data: dict[str, Any]) -> TeamPulseObservation:
    """Strictly validate an observation dictionary for team pulse."""
    if not isinstance(data, dict):
        raise ValueError("TeamPulseObservation must be a dict")
    allowed = {"focus_areas", "friction_categories", "momentum", "confidence"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in TeamPulseObservation: {sorted(unknown)}")

    raw_focus = data.get("focus_areas", [])
    if not isinstance(raw_focus, list):
        raise ValueError("focus_areas must be a list")
    if len(raw_focus) > 5:
        raise ValueError("focus_areas exceeds limit of 5 items")
    focus = [sanitize_topic_label(f) for f in raw_focus]

    raw_frictions = data.get("friction_categories", [])
    if not isinstance(raw_frictions, list):
        raise ValueError("friction_categories must be a list")
    for fc in raw_frictions:
        if fc not in ALLOWLISTED_FRICTIONS:
            raise ValueError("Invalid friction category in observation")

    momentum = data.get("momentum", "unknown")
    if momentum not in MOMENTUM_BANDS:
        raise ValueError("Invalid momentum in observation")

    conf = _validate_confidence_positive(data.get("confidence", 0.5), "confidence")

    if not has_team_pulse_observation_fields(data):
        raise ValueError("TeamPulseObservation carries no observed fields")

    return TeamPulseObservation(
        focus_areas=focus,
        friction_categories=list(raw_frictions),
        momentum=momentum,
        confidence=conf,
    )


def validate_agent_calibration_observation(data: dict[str, Any]) -> AgentCalibrationObservation:
    """Strictly validate an observation dictionary for agent calibration."""
    if not isinstance(data, dict):
        raise ValueError("AgentCalibrationObservation must be a dict")
    allowed = {"preferred_verbosity", "preferred_directness", "formatting_avoidances", "confidence"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in AgentCalibrationObservation: {sorted(unknown)}")

    verbosity = data.get("preferred_verbosity", "concise")
    if verbosity not in VERBOSITY_BANDS:
        raise ValueError("Invalid preferred_verbosity in observation")

    directness = data.get("preferred_directness", "direct")
    if directness not in DIRECTNESS_BANDS:
        raise ValueError("Invalid preferred_directness in observation")

    raw_avoidances = data.get("formatting_avoidances", [])
    if not isinstance(raw_avoidances, list):
        raise ValueError("formatting_avoidances must be a list")
    for fa in raw_avoidances:
        if fa not in ALLOWLISTED_FORMATTING_AVOIDANCES:
            raise ValueError("Invalid formatting avoidance in observation")

    conf = _validate_confidence_positive(data.get("confidence", 0.5), "confidence")

    if not has_agent_calibration_observation_fields(data):
        raise ValueError("AgentCalibrationObservation carries no observed fields")

    return AgentCalibrationObservation(
        preferred_verbosity=verbosity,
        preferred_directness=directness,
        formatting_avoidances=list(raw_avoidances),
        confidence=conf,
    )


def _lru_merge(existing: list[str], incoming: list[str], bound: int) -> list[str]:
    """Merge short labels LRU-by-re-observation (audit F11).

    A re-observed label moves to the most-recent position; when over the bound,
    labels are evicted from the stalest (least recently observed) position.
    The same topic-label policy applies at merge time (GAP 2): stored labels
    that fail it are deterministically evicted now-invalid labels.
    """
    merged: list[str] = []
    for label in existing:
        normalized = normalize_valid_topic_label(label)
        if normalized is not None and normalized not in merged:
            merged.append(normalized)
    for label in incoming:
        normalized = normalize_valid_topic_label(label)
        if normalized is None:
            continue
        if normalized in merged:
            merged.remove(normalized)
        merged.append(normalized)
    return merged[-bound:]


def transient_from_observation(
    obs: MemberObservation, now: int,
) -> TransientMemberState | None:
    """Build expiring transient state from an observation, or None when absent."""
    if obs.energy is None and not obs.current_focus:
        return None
    return TransientMemberState(
        energy=obs.energy or "unknown",
        current_focus=obs.current_focus[:3],
        observed_at=now,
        expires_at=now + TRANSIENT_TTL_SECONDS,
        schema_version=MEMORY_SCHEMA_VERSION,
    )


def merge_member_observation(
    current: MemberProfile | None,
    obs: MemberObservation,
    now: int,
) -> tuple[MemberProfile, TransientMemberState | None]:
    """Deterministically merge a member observation into stable profile and transient state.

    Rules (plan Phase 2.4/2.5, audit F2/F3/F11, GAP 2/3/4):
    - Fresh profiles carry ONLY the observed dimensions (GAP 3): unobserved
      categorical preferences stay None with confidence 0.0 and evidence 0;
      directness stays None when unobserved. Nothing is fabricated.
    - Numeric updates are weighted by observation confidence against the existing
      confidence *decayed by recency* (30-day half-life), with per-observation
      movement bounded by MAX_NUMERIC_SHIFT and clamped to [0, 1]; an unobserved
      directness adopts the observation value.
    - Categorical preferences change slowly (GAP 4): a single contradiction
      NEVER flips a value.
      (a) no current value (unobserved) -> adopt the observation;
      (b) agreement -> reinforce confidence (bounded);
      (c) contradiction -> reduce the current field's confidence by a bounded
          step scaled by observation confidence, never flip in this step;
      (d) flip only when the current field's effective confidence has already
          decayed below CATEGORICAL_FLIP_CONF_FLOOR AND the observation
          confidence exceeds CATEGORICAL_FLIP_MIN_CONFIDENCE.
    - Confidence changes are clamped to [0, 1] on every path.
    - Recurring topics merge LRU-by-re-observation, evict from the stalest
      position beyond MAX_RECURRING_TOPICS, and stored labels failing the
      topic-label policy are deterministically dropped (GAP 2).
    - Transient energy and current_focus expire after TRANSIENT_TTL_SECONDS.
    """
    # Defense-in-depth: both public boundaries validate non-finite values, but
    # the merge itself must never propagate NaN/inf arithmetic into a profile.
    if not math.isfinite(obs.confidence):
        raise ValueError("observation confidence must be finite")
    if obs.directness is not None and not math.isfinite(obs.directness):
        raise ValueError("observation directness must be finite")
    if current is None:
        # Initialize a fresh profile from ONLY the observed fields (GAP 3):
        # unobserved dimensions are stored as None with confidence 0.0 and
        # evidence 0 instead of fabricated defaults.
        comm = CommunicationProfile(
            directness=obs.directness,
            detail_preference=obs.detail_preference,
            challenge_preference=obs.challenge_preference,
            humor_preference=obs.humor_preference,
            decision_style=obs.decision_style,
        )
        observed = {
            "directness": obs.directness is not None,
            "detail_preference": obs.detail_preference is not None,
            "challenge_preference": obs.challenge_preference is not None,
            "humor_preference": obs.humor_preference is not None,
            "decision_style": obs.decision_style is not None,
        }
        conf = {
            k: (round(obs.confidence, 3) if observed[k] else 0.0)
            for k in COMMUNICATION_KEYS
        }
        ev = {k: (1 if observed[k] else 0) for k in COMMUNICATION_KEYS}
        last_obs = {k: now for k in COMMUNICATION_KEYS if observed[k]}
        merged_topics = [
            label
            for label in (
                normalize_valid_topic_label(t) for t in obs.topics
            )
            if label is not None
        ]
        profile = MemberProfile(
            schema_version=MEMORY_SCHEMA_VERSION,
            communication=comm,
            confidence=conf,
            evidence_count=ev,
            last_observed_at=last_obs,
            recurring_topics=merged_topics[:MAX_RECURRING_TOPICS],
            created_at=now,
            updated_at=now,
            last_interaction_at=now,
        )
    else:
        # Update numeric directness gradually, weighting the existing evidence by
        # its recency so stale profiles move faster than fresh ones (audit F2).
        cur_dir = current.communication.directness
        cur_dir_conf = current.confidence.get("directness", 0.0)
        cur_dir_ev = current.evidence_count.get("directness", 0)
        if obs.directness is not None:
            if cur_dir is None or cur_dir_ev <= 0 or cur_dir_conf <= 0.0:
                # Unobserved numeric dimension -> adopt the observed value (GAP 3).
                new_dir = obs.directness
                new_dir_conf = round(obs.confidence, 3)
                new_dir_ev = 1
                dir_obs_time = now
            else:
                recency = recency_factor(
                    now,
                    current.last_observed_at.get("directness", now),
                    PROFILE_RECENCY_HALF_LIFE_SECONDS,
                )
                effective_cur_conf = cur_dir_conf * recency
                raw_delta = obs.directness - cur_dir
                weight = obs.confidence / (
                    effective_cur_conf + obs.confidence + NUMERIC_WEIGHT_SMOOTHING
                )
                bounded_delta = max(
                    -MAX_NUMERIC_SHIFT, min(MAX_NUMERIC_SHIFT, raw_delta * weight)
                )
                new_dir = round(max(0.0, min(1.0, cur_dir + bounded_delta)), 3)
                new_dir_conf = round(
                    max(
                        0.0,
                        min(1.0, effective_cur_conf * 0.95 + obs.confidence * 0.15),
                    ),
                    3,
                )
                new_dir_ev = cur_dir_ev + 1
                dir_obs_time = now
        else:
            new_dir = cur_dir
            new_dir_conf = cur_dir_conf
            new_dir_ev = cur_dir_ev
            dir_obs_time = current.last_observed_at.get("directness", now)

        # Update categorical fields (audit F3, GAP 4).
        def update_cat(
            field_name: str,
            cur_val: Any,
            obs_val: Any,
        ) -> tuple[Any, float, int, int]:
            c_conf = current.confidence.get(field_name, 0.0)
            c_ev = current.evidence_count.get(field_name, 0)
            c_obs = current.last_observed_at.get(field_name, now)
            if obs_val is None:
                return cur_val, c_conf, c_ev, c_obs
            if cur_val is None or c_ev <= 0 or c_conf <= 0.0:
                # (a) No current value (unobserved) -> adopt the observed value.
                return obs_val, round(obs.confidence, 3), 1, now
            recency = recency_factor(now, c_obs, PROFILE_RECENCY_HALF_LIFE_SECONDS)
            effective_conf = c_conf * recency
            if obs_val == cur_val:
                # (b) Consistent observation -> reinforce confidence (bounded).
                gain = CONSISTENT_CONF_GAIN * obs.confidence
                new_conf = round(max(0.0, min(1.0, effective_conf + gain)), 3)
                return cur_val, new_conf, c_ev + 1, now
            # (c) Contradiction -> reduce the current field's confidence by a
            # bounded step scaled by observation confidence; NEVER flip in this
            # step regardless of prior evidence volume (GAP 4).
            reduced_conf = round(
                max(
                    CONTRADICTION_CONF_FLOOR,
                    effective_conf - CONTRADICTION_CONF_PENALTY * obs.confidence,
                ),
                3,
            )
            if (
                effective_conf < CATEGORICAL_FLIP_CONF_FLOOR
                and obs.confidence > CATEGORICAL_FLIP_MIN_CONFIDENCE
            ):
                # (d) The current value is already worn down below the flip
                # floor and this contradiction is strong -> flip with bounded
                # confidence.
                return (
                    obs_val,
                    round(obs.confidence * FLIP_CONFIDENCE_FACTOR, 3),
                    c_ev + 1,
                    now,
                )
            return cur_val, reduced_conf, c_ev + 1, now

        new_detail, detail_conf, detail_ev, detail_obs = update_cat(
            "detail_preference",
            current.communication.detail_preference,
            obs.detail_preference,
        )
        new_challenge, chal_conf, chal_ev, chal_obs = update_cat(
            "challenge_preference",
            current.communication.challenge_preference,
            obs.challenge_preference,
        )
        new_humor, humor_conf, humor_ev, humor_obs = update_cat(
            "humor_preference",
            current.communication.humor_preference,
            obs.humor_preference,
        )
        new_decision, dec_conf, dec_ev, dec_obs = update_cat(
            "decision_style",
            current.communication.decision_style,
            obs.decision_style,
        )

        new_comm = CommunicationProfile(
            directness=new_dir,
            detail_preference=new_detail,
            challenge_preference=new_challenge,
            humor_preference=new_humor,
            decision_style=new_decision,
        )
        new_conf = {
            "directness": new_dir_conf,
            "detail_preference": detail_conf,
            "challenge_preference": chal_conf,
            "humor_preference": humor_conf,
            "decision_style": dec_conf,
        }
        new_ev = {
            "directness": new_dir_ev,
            "detail_preference": detail_ev,
            "challenge_preference": chal_ev,
            "humor_preference": humor_ev,
            "decision_style": dec_ev,
        }
        new_last_obs = {
            "directness": dir_obs_time,
            "detail_preference": detail_obs,
            "challenge_preference": chal_obs,
            "humor_preference": humor_obs,
            "decision_style": dec_obs,
        }

        combined_topics = _lru_merge(
            current.recurring_topics, obs.topics, MAX_RECURRING_TOPICS
        )

        profile = MemberProfile(
            schema_version=MEMORY_SCHEMA_VERSION,
            communication=new_comm,
            confidence=new_conf,
            evidence_count=new_ev,
            last_observed_at=new_last_obs,
            recurring_topics=combined_topics,
            created_at=current.created_at,
            updated_at=now,
            last_interaction_at=now,
        )

    transient = transient_from_observation(obs, now)
    return profile, transient


def merge_team_pulse_observation(
    current: TeamPulse | None,
    obs: TeamPulseObservation,
    now: int,
) -> TeamPulse:
    """Deterministically merge a team pulse observation.

    Pulse records hard-expire after PULSE_TTL_SECONDS; within the window the
    existing record is downweighted by a recency factor with a half-life of half
    the TTL (audit F2), and confidence is clamped to [0, 1].
    """
    if current is None or current.expires_at <= now:
        return TeamPulse(
            schema_version=MEMORY_SCHEMA_VERSION,
            focus_areas=obs.focus_areas[:5],
            friction_categories=sorted(set(obs.friction_categories)),
            momentum=obs.momentum,
            confidence=round(obs.confidence, 3),
            evidence_count=1,
            observed_at=now,
            expires_at=now + PULSE_TTL_SECONDS,
        )

    recency = recency_factor(now, current.observed_at, PULSE_RECENCY_HALF_LIFE_SECONDS)
    combined_focus = _lru_merge(current.focus_areas, obs.focus_areas, 5)
    combined_frictions = sorted(
        set(current.friction_categories) | set(obs.friction_categories)
    )

    momentum = obs.momentum if obs.momentum != "unknown" else current.momentum
    new_conf = round(
        max(0.0, min(1.0, current.confidence * 0.8 * recency + obs.confidence * 0.2)), 3
    )
    new_ev = current.evidence_count + 1

    return TeamPulse(
        schema_version=MEMORY_SCHEMA_VERSION,
        focus_areas=combined_focus,
        friction_categories=combined_frictions,
        momentum=momentum,
        confidence=new_conf,
        evidence_count=new_ev,
        observed_at=now,
        expires_at=now + PULSE_TTL_SECONDS,
    )


def merge_agent_calibration_observation(
    current: AgentCalibration | None,
    obs: AgentCalibrationObservation,
    now: int,
) -> AgentCalibration:
    """Deterministically merge an agent calibration observation.

    Calibration records hard-expire after CALIBRATION_TTL_SECONDS; within the
    window the existing record is downweighted by a recency factor with a
    half-life of half the TTL (audit F2), and confidence is clamped to [0, 1].
    """
    if current is None or current.expires_at <= now:
        return AgentCalibration(
            schema_version=MEMORY_SCHEMA_VERSION,
            preferred_verbosity=obs.preferred_verbosity,
            preferred_directness=obs.preferred_directness,
            formatting_avoidances=sorted(set(obs.formatting_avoidances)),
            confidence=round(obs.confidence, 3),
            evidence_count=1,
            updated_at=now,
            expires_at=now + CALIBRATION_TTL_SECONDS,
        )

    recency = recency_factor(
        now, current.updated_at, CALIBRATION_RECENCY_HALF_LIFE_SECONDS
    )
    combined_avoidances = sorted(
        set(current.formatting_avoidances) | set(obs.formatting_avoidances)
    )
    verbosity = obs.preferred_verbosity if obs.confidence >= 0.5 else current.preferred_verbosity
    directness = (
        obs.preferred_directness if obs.confidence >= 0.5 else current.preferred_directness
    )
    blended = current.confidence * 0.85 * recency + obs.confidence * 0.15
    new_conf = round(max(0.0, min(1.0, blended)), 3)
    new_ev = current.evidence_count + 1

    return AgentCalibration(
        schema_version=MEMORY_SCHEMA_VERSION,
        preferred_verbosity=verbosity,
        preferred_directness=directness,
        formatting_avoidances=combined_avoidances,
        confidence=new_conf,
        evidence_count=new_ev,
        updated_at=now,
        expires_at=now + CALIBRATION_TTL_SECONDS,
    )


# --- Safe prompt-rendering views (pinned contract PIN-1) -------------------------
#
# Frozen, JSON-native views over stored memory. Every value is render-safe:
# allowlisted enum strings, band strings, sanitized short labels, numbers, or
# None. Anything whose confidence is below CONFIDENCE_FLOOR becomes None/absent.


@dataclass(frozen=True)
class ProfileView:
    """Render-safe view of one member's stable profile."""

    member_id: str
    handle: str
    directness_band: str | None
    detail_preference: str | None
    challenge_preference: str | None
    humor_preference: str | None
    decision_style: str | None
    recurring_topics: tuple[str, ...]
    confidence: float  # overall: mean of above-floor field confidences, 0.0 if none


@dataclass(frozen=True)
class TransientView:
    """Render-safe view of one member's unexpired transient state."""

    member_id: str
    handle: str
    energy: str | None  # allowlisted enum or None
    current_focus: tuple[str, ...]


@dataclass(frozen=True)
class TeamPulseView:
    """Render-safe view of one scope's unexpired team pulse."""

    focus_areas: tuple[str, ...]
    friction_categories: tuple[str, ...]
    momentum: str
    confidence: float


@dataclass(frozen=True)
class CalibrationView:
    """Render-safe view of one scope's unexpired agent calibration."""

    verbosity_band: str | None
    directness_band: str | None
    formatting_avoidances: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class MemorySnapshot:
    """Render-safe, scope-isolated memory snapshot for prompt rendering."""

    profiles: tuple[ProfileView, ...]
    transient: tuple[TransientView, ...]
    team_pulse: TeamPulseView | None
    calibration: CalibrationView | None
    generated_at: str  # ISO-8601 UTC

    @property
    def is_empty(self) -> bool:
        return (
            not self.profiles
            and not self.transient
            and self.team_pulse is None
            and self.calibration is None
        )


def _safe_labels(labels: list[str]) -> tuple[str, ...]:
    """Return sanitized labels, silently dropping anything that fails sanitization."""
    safe: list[str] = []
    for label in labels:
        try:
            safe.append(sanitize_label(label, max_len=40))
        except ValueError:
            continue
    return tuple(safe)


def _safe_handle(handle: Any) -> str:
    try:
        return sanitize_label(handle or "member", max_len=64)
    except ValueError:
        return "member"


def build_profile_view(
    member_id: str, handle: str, profile: MemberProfile,
) -> ProfileView | None:
    """Build a render-safe profile view, or None when nothing clears the floor.

    A profile contributes only when its overall confidence (mean of above-floor
    field confidences) is at least CONFIDENCE_FLOOR and at least one field is
    render-ready.
    """
    field_names = (
        "directness",
        "detail_preference",
        "challenge_preference",
        "humor_preference",
        "decision_style",
    )
    above_floor = [
        profile.confidence[k]
        for k in field_names
        if profile.confidence.get(k, 0.0) >= CONFIDENCE_FLOOR
    ]
    if not above_floor:
        return None
    overall = round(sum(above_floor) / len(above_floor), 3)
    if overall < CONFIDENCE_FLOOR:
        return None

    def band_or_none(numeric: float | None, field: str) -> str | None:
        if numeric is None or profile.confidence.get(field, 0.0) < CONFIDENCE_FLOOR:
            return None
        try:
            return directness_band(numeric)
        except ValueError:
            return None

    def enum_or_none(value: str, field: str) -> str | None:
        if profile.confidence.get(field, 0.0) < CONFIDENCE_FLOOR:
            return None
        return value

    return ProfileView(
        member_id=str(member_id),
        handle=_safe_handle(handle),
        directness_band=band_or_none(profile.communication.directness, "directness"),
        detail_preference=enum_or_none(
            profile.communication.detail_preference, "detail_preference"
        ),
        challenge_preference=enum_or_none(
            profile.communication.challenge_preference, "challenge_preference"
        ),
        humor_preference=enum_or_none(
            profile.communication.humor_preference, "humor_preference"
        ),
        decision_style=enum_or_none(
            profile.communication.decision_style, "decision_style"
        ),
        recurring_topics=_safe_labels(profile.recurring_topics),
        confidence=overall,
    )


def build_transient_view(
    member_id: str, handle: str, state: TransientMemberState
) -> TransientView | None:
    """Build a render-safe transient view, or None when there is nothing to render."""
    energy = state.energy if state.energy in ENERGY_BANDS else None
    focus = _safe_labels(state.current_focus)
    if energy is None and not focus:
        return None
    return TransientView(
        member_id=str(member_id),
        handle=_safe_handle(handle),
        energy=energy,
        current_focus=focus,
    )


def build_team_pulse_view(pulse: TeamPulse) -> TeamPulseView | None:
    """Build a render-safe team pulse view, or None when there is nothing usable."""
    focus = _safe_labels(pulse.focus_areas)
    frictions = tuple(
        f for f in pulse.friction_categories if f in ALLOWLISTED_FRICTIONS
    )
    if not focus and not frictions and pulse.momentum == "unknown":
        return None
    return TeamPulseView(
        focus_areas=focus,
        friction_categories=frictions,
        momentum=pulse.momentum,
        confidence=round(pulse.confidence, 3),
    )


def build_calibration_view(calibration: AgentCalibration) -> CalibrationView | None:
    """Build a render-safe calibration view, or None when confidence is below the floor."""
    if calibration.confidence < CONFIDENCE_FLOOR:
        return None
    return CalibrationView(
        verbosity_band=calibration.preferred_verbosity,
        directness_band=calibration.preferred_directness,
        formatting_avoidances=tuple(
            f
            for f in calibration.formatting_avoidances
            if f in ALLOWLISTED_FORMATTING_AVOIDANCES
        ),
        confidence=round(calibration.confidence, 3),
    )
