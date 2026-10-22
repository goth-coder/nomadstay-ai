"""
Canonical aspects a digital nomad asks for, and how to detect them.

- amenity: regex over the host's amenity list → a listing *claim*
- keywords: multilingual regex over free text (reviews, descriptions),
  used to keep rare mentions when sampling reviews. It only finds mentions;
  whether a review supports or contradicts the aspect is decided later.
- queries: texts for the semantic search of evidence

Patterns come from the amenity names seen in the Porto snapshot (2026-09-19).
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Aspect:
    name: str
    label: str
    amenity: re.Pattern | None
    keywords: re.Pattern
    queries: list[str] = field(default_factory=list)


def words(*terms: str) -> re.Pattern:
    # Whole words/phrases, any case, accents included
    return re.compile(r"(?<!\w)(?:" + "|".join(terms) + r")(?!\w)", re.IGNORECASE)


ASPECTS: dict[str, Aspect] = {
    aspect.name: aspect
    for aspect in [
        Aspect(
            name="wifi",
            label="Reliable Wi-Fi",
            # "Wifi", "Fast wifi – 300 Mbps", "Wifi – 500 Mbps", "Pocket wifi", "Ethernet connection"
            amenity=re.compile(r"^(?:fast |pocket )?wifi\b|^ethernet connection$", re.IGNORECASE),
            keywords=words(
                r"wi-?fi", r"wi fi", r"internet", r"mbps", r"router",
                r"connection", r"connexion", r"conexi[oó]n", r"conex[aã]o",
                r"verbindung", r"wlan",
            ),
            queries=[
                "fast and stable wifi, good for video calls",
                "the internet was slow or kept disconnecting",
            ],
        ),
        Aspect(
            name="workspace",
            label="Workspace",
            amenity=re.compile(r"^(?:dedicated|laptop friendly) workspace$", re.IGNORECASE),
            keywords=words(
                r"desk", r"workspace", r"work from", r"remote work\w*", r"working remotely",
                r"home office", r"zoom", r"video calls?", r"secret[aá]ria",
                r"escrit[oó]rio", r"bureau", r"escritorio", r"schreibtisch",
                r"trabalhar", r"travailler", r"trabajar", r"arbeiten",
            ),
            queries=[
                "good desk and chair to work on a laptop all day",
                "no proper place to work, had to work on the bed or sofa",
            ],
        ),
        Aspect(
            name="quiet",
            label="Quiet",
            amenity=None,  # no amenity claims silence ("Sound system" is a speaker)
            keywords=words(
                r"quiet", r"nois[ey]", r"noise", r"loud", r"silent",
                r"ru[ií]do", r"barulh\w*", r"silencios[oa]", r"sossegad[oa]",
                r"calme", r"bruit", r"bruyant\w*", r"ruidos[oa]", r"tranquil[oa]",
                r"l[äa]rm", r"leise", r"ruhig\w*",
            ),
            queries=[
                "very quiet apartment, slept well, peaceful to work",
                "noisy at night, street noise and bars, thin walls",
            ],
        ),
        Aspect(
            name="gym_nearby",
            label="Gym nearby",
            # "Gym", "Shared gym in building", "Private gym in building", "Shared gym nearby"
            amenity=re.compile(r"\bgym\b", re.IGNORECASE),
            keywords=words(
                r"gym", r"gyms", r"fitness", r"gin[aá]sio", r"academia", r"gimnasio",
                r"salle de sport", r"fitnessstudio", r"crossfit",
            ),
            queries=[
                "there is a gym close to the apartment",
                "gym around the corner, easy to work out",
            ],
        ),
        Aspect(
            name="transport",
            label="Public transport",
            amenity=None,
            keywords=words(
                r"metro", r"subway", r"bus", r"tram", r"train", r"station",
                r"esta[cç][aã]o", r"autocarro", r"el[eé]trico", r"m[ée]tro",
                r"gare", r"estaci[oó]n", r"autob[uú]s", r"u-bahn", r"bahnhof",
            ),
            queries=[
                "metro station a few minutes walk, easy to get around",
                "far from public transport, needed taxis",
            ],
        ),
    ]
}

WIFI_MBPS = re.compile(r"wifi\W+(\d+)\s*mbps", re.IGNORECASE)


def claimedAspects(amenities: list[str]) -> dict[str, str]:
    """
    Aspect -> the first raw amenity that claims it.
    Only the host's amenity list counts as a claim.
    """
    claims = {}
    for name, aspect in ASPECTS.items():
        if aspect.amenity is None:
            continue
        match = next((a for a in amenities if aspect.amenity.search(a.strip())), None)
        if match:
            claims[name] = match.strip()
    return claims


def parseWifiMbps(amenities: list[str]) -> int | None:
    # Highest speed listed; None when the host listed no speed test
    speeds = [int(m.group(1)) for a in amenities if (m := WIFI_MBPS.search(a))]
    return max(speeds, default=None)


def mentionedAspects(text: str) -> set[str]:
    return {name for name, aspect in ASPECTS.items() if aspect.keywords.search(text)}
