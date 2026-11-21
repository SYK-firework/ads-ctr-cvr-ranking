"""Feast definitions: one entity (campaign) and one feature view of ad-side features."""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

campaign = Entity(name="campaign", join_keys=["campaign"], value_type=ValueType.INT64,
                  description="Criteo Attribution campaign id")

source = FileSource(
    path=str(Path(__file__).resolve().parent / "data" / "campaign_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

campaign_features = FeatureView(
    name="campaign_features",
    entities=[campaign],
    ttl=timedelta(days=3650),          # offline timestamps are synthetic (2026-01-01 + data seconds)
    schema=[Field(name="cpo", dtype=Float32),
            Field(name="campaign_impressions", dtype=Int64),
            Field(name="campaign_clicks", dtype=Int64)],
    source=source,
    online=True,
)
