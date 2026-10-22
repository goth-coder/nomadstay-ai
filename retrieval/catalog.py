"""
Catalog: the deterministic side of the search. City, budget and must-have
filters run here, in code, never in the LLM.

A must-have is checked against listing claims (amenities). For gym_nearby the
map also counts: a mapped gym within the radius satisfies it, since the user
asks for a gym near the stay, not necessarily inside it.
"""

from dataclasses import dataclass

import pandas as pd

CITY_ALIASES = {
    "porto": {"porto", "oporto", "o porto", "porto, portugal", "porto portugal"},
}
DEFAULT_GYM_RADIUS_M = 500


def normalizeCity(city: str | None) -> str | None:
    """'Oporto' / 'PORTO' -> 'porto'; unknown cities come back lowercased."""
    if city is None:
        return None
    key = " ".join(city.strip().lower().split())
    for canonical, aliases in CITY_ALIASES.items():
        if key == canonical or key in aliases:
            return canonical
    return key


@dataclass
class Catalog:
    listings: pd.DataFrame  # one row per Listing (claims as dict, amenities as list)
    city: str
    currency: str
    snapshot_date: str

    def filter(
        self,
        max_price: float | None = None,
        must_have: list[str] | None = None,
        max_gym_distance_m: float | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Listings that satisfy every hard constraint, plus counts that explain
        what each constraint removed (used to suggest what to relax).
        Listings without a price never pass a budget filter.
        """
        frame = self.listings
        must_have = list(dict.fromkeys(must_have or []))
        gym_radius = DEFAULT_GYM_RADIUS_M if max_gym_distance_m is None else max_gym_distance_m

        price_ok = pd.Series(True, index=frame.index)
        if max_price is not None:
            price_ok = frame["price"].notna() & (frame["price"] <= max_price)

        needs_ok = pd.Series(True, index=frame.index)
        for aspect in must_have:
            claimed = frame["claims"].map(lambda claims, a=aspect: a in claims)
            if aspect == "gym_nearby":
                claimed = claimed | (frame["nearest_gym_m"].fillna(float("inf")) <= gym_radius)
            needs_ok &= claimed

        gym_ok = pd.Series(True, index=frame.index)
        if max_gym_distance_m is not None:
            gym_ok = frame["nearest_gym_m"].fillna(float("inf")) <= max_gym_distance_m

        kept = frame[price_ok & needs_ok & gym_ok]
        counts = {
            "in_city": len(frame),
            "matched": len(kept),
            "without_price": int(frame["price"].isna().sum()) if max_price is not None else 0,
            # how many would match if one constraint were dropped
            "without_budget_limit": int((needs_ok & gym_ok).sum()),
            "without_must_have": int((price_ok & gym_ok).sum()),
            "without_gym_distance": int((price_ok & needs_ok).sum()),
        }
        return kept, counts

    def get(self, name_or_id: str) -> pd.DataFrame:
        """Listings by exact id or by name (case-insensitive, may match several)."""
        by_id = self.listings[self.listings["listing_id"] == name_or_id]
        if not by_id.empty:
            return by_id
        needle = name_or_id.strip().lower()
        return self.listings[self.listings["name"].str.lower().str.contains(needle, regex=False)]
