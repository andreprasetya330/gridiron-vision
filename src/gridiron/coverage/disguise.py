"""Pre-snap look versus post-snap coverage.

The overlay's single coverage call is what they ran. Disguise is the join of
that call with what they showed: the geometric shell at the snap, and (when a
pre-snap model exists) the man/zone family of the look.

A Cover 1 vs Cover 3 disagreement is not a disguise — those two looks are the
same 1-high picture. A disguise is man vs zone, or a 2-high picture that is
gone by 2.5 seconds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from gridiron.cues.vocabulary import DEEP_THRESHOLD_YD, extract_cues
from gridiron.taxonomy import is_man, shell_of
from gridiron.tracking.schema import PlayTracks

DisguiseKind = Literal["none", "uncertain", "shell", "family", "both"]

FAMILY_MIN_PRE = 0.40
FAMILY_MIN_POST = 0.45
SHELL_LATE_T = 2.5


@dataclass
class DisguiseCall:
    showed: str | None
    ran: str
    showed_shell: str | None
    ran_shell: str | None
    kind: DisguiseKind
    disguised: bool
    family_mismatch: bool
    shell_mismatch: bool
    showed_confidence: float | None = None
    ran_confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def geometric_shell(play: PlayTracks, seconds: float, threshold: float = DEEP_THRESHOLD_YD) -> str | None:
    """How many defenders are playing deep at `seconds`, same rule as the cue vocabulary."""
    deep = 0
    seen = 0
    for defender in play.defense:
        x, _y = defender.at(seconds)
        if not _finite(x):
            continue
        seen += 1
        if x >= threshold:
            deep += 1
    if seen == 0:
        return None
    return {0: "0-high", 1: "1-high", 2: "2-high"}.get(deep, "3+-high")


def _finite(value: float) -> bool:
    return value == value  # NaN != NaN


def classify_disguise(
    play: PlayTracks,
    post: dict[str, Any],
    pre: dict[str, Any] | None = None,
) -> DisguiseCall:
    ran = str(post["coverage"])
    ran_conf = float(post.get("confidence") or 0.0)
    showed_shell = geometric_shell(play, 0.0)
    ran_shell = geometric_shell(play, SHELL_LATE_T)
    if showed_shell is None:
        cues = extract_cues(play)
        showed_shell = cues.categorical.get("shell")

    shell_mismatch = _shell_is_disguise(showed_shell, ran_shell)
    if play.quality.defenders_detected < 10:
        # Missing safeties make snap vs 2.5s deep-counts jump around. Do not
        # call a shell disguise on a play that never saw the secondary.
        shell_mismatch = False

    family_mismatch = False
    showed: str | None = None
    showed_conf: float | None = None
    uncertain = False
    if pre and pre.get("coverage"):
        showed = str(pre["coverage"])
        showed_conf = float(pre.get("confidence") or 0.0)
        if showed_conf >= FAMILY_MIN_PRE and ran_conf >= FAMILY_MIN_POST:
            raw_family = is_man(showed) != is_man(ran)
            # Cover 1 vs Cover 3 from a 1-high picture is the model's hardest pair,
            # not a disguise a coach should trust. Require a 2-high (or empty)
            # look before calling man/zone a disguise.
            family_mismatch = raw_family and showed_shell in {"0-high", "2-high", "3+-high"}
        elif showed != ran:
            uncertain = True

    if family_mismatch and shell_mismatch:
        kind: DisguiseKind = "both"
    elif family_mismatch:
        kind = "family"
    elif shell_mismatch:
        kind = "shell"
    elif uncertain:
        kind = "uncertain"
    else:
        kind = "none"

    return DisguiseCall(
        showed=showed or showed_shell,
        ran=ran,
        showed_shell=showed_shell,
        ran_shell=ran_shell or shell_of(ran),
        kind=kind,
        disguised=kind in {"family", "shell", "both"},
        family_mismatch=family_mismatch,
        shell_mismatch=shell_mismatch,
        showed_confidence=showed_conf,
        ran_confidence=ran_conf,
    )


def _shell_is_disguise(showed: str | None, ran: str | None) -> bool:
    if not showed or not ran or showed == ran:
        return False
    # 2-high look that collapses or turns into three-deep is the textbook rotate.
    if showed == "2-high" and ran in {"0-high", "1-high", "3+-high"}:
        return True
    # Press/zero look that becomes two-high after the snap.
    if showed in {"0-high", "1-high"} and ran == "2-high":
        return True
    return False


def annotate_predictions(
    plays: Iterable[PlayTracks],
    post_predictions: list[dict[str, Any]],
    pre_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach look / disguise fields onto the overlay payload."""
    plays_by_id = {p.play_id: p for p in plays}
    pre_by_id = {p["play_id"]: p for p in pre_predictions or []}
    out: list[dict[str, Any]] = []
    for post in post_predictions:
        play = plays_by_id.get(post["play_id"])
        if play is None:
            out.append(post)
            continue
        pre = pre_by_id.get(post["play_id"])
        call = classify_disguise(play, post, pre)
        row = dict(post)
        if pre:
            row["presnap"] = {
                "coverage": pre.get("coverage"),
                "confidence": pre.get("confidence"),
                "probabilities": pre.get("probabilities", {}),
                "runner_up": pre.get("runner_up"),
            }
        row["disguise"] = call.as_dict()
        out.append(row)
    return out


def summarize(predictions: list[dict[str, Any]], team: str | None = None) -> dict[str, Any]:
    rows = [p for p in predictions if p.get("disguise")]
    if team:
        # Team is on the play, not always on the prediction. Callers that have
        # it stashed under defense_team can filter first.
        rows = [p for p in rows if p.get("defense_team") == team]
    n = len(rows)
    counts = {"none": 0, "uncertain": 0, "shell": 0, "family": 0, "both": 0}
    for row in rows:
        kind = str(row["disguise"].get("kind") or "none")
        counts[kind] = counts.get(kind, 0) + 1
    disguised = counts["shell"] + counts["family"] + counts["both"]
    return {
        "n": n,
        "disguised": disguised,
        "rate": (disguised / n) if n else 0.0,
        "kinds": counts,
    }
