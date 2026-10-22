"""
Data contract of the NomadStay artifacts (v2).

Listing: one normalized row of listings.parquet (built from Inside Airbnb).
Chunk: one row of chunks.parquet; its position is the FAISS id.

Listing claims (amenities, description) and guest evidence (reviews) are kept
apart: a Chunk says where its text came from through `source`.
"""

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PriceStatus = Literal["listed", "quote", "missing"]


class Listing(BaseModel):
    listing_id: str  # Inside Airbnb ids mix old numeric and 19-digit ids
    name: str
    url: str
    city: str  # normalized key, e.g. "porto"
    neighbourhood: str | None = None
    neighbourhood_group: str | None = None
    latitude: float
    longitude: float
    property_type: str | None = None
    room_type: str | None = None
    accommodates: int | None = None

    # Snapshot price per night, never a live price
    price: float | None = Field(default=None, gt=0)
    currency: str
    price_status: PriceStatus

    amenities: list[str] = []  # raw, as listed by the host
    claims: dict[str, str] = {}  # aspect -> raw amenity that claims it
    wifi_mbps: int | None = None  # host speed test listed as an amenity

    # Straight-line distance to the nearest mapped gym (OpenStreetMap).
    # Airbnb shifts coordinates up to ~150 m, so it is approximate.
    nearest_gym_m: float | None = Field(default=None, ge=0)
    nearest_gym_name: str | None = None

    rating: float | None = Field(default=None, ge=0, le=5)
    number_of_reviews: int = 0
    last_review: dt.date | None = None

    description: str = ""
    neighborhood_overview: str | None = None
    snapshot_date: dt.date

    @model_validator(mode="after")
    def priceMatchesStatus(self):
        if (self.price is None) != (self.price_status == "missing"):
            raise ValueError("price must be set unless price_status is 'missing'")
        return self


class Chunk(BaseModel):
    chunk_id: int  # == FAISS position
    listing_id: str
    source: Literal["listing", "review"]
    field: Literal["description", "neighborhood_overview", "comments"]
    review_id: str | None = None
    date: dt.date | None = None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def reviewHasId(self):
        is_review = self.source == "review"
        if is_review != (self.field == "comments"):
            raise ValueError("review chunks come from comments, listing chunks do not")
        if is_review != (self.review_id is not None):
            raise ValueError("review_id is required for review chunks only")
        return self
