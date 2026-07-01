"""Database models and session configuration for the House Pricing API."""
import datetime
import os

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Reads the database URL from the environment (set by Docker Compose).
# Falls back to a local SQLite file when running outside of Docker.
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./house_pricing_db.db"
)

# SQLite needs an extra connection argument; Postgres does not.
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    """Stores every prediction request/response for auditing purposes."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    # User-provided inputs
    date = Column(String)
    bedrooms = Column(Float)
    bathrooms = Column(Float)
    sqft_living = Column(Float)
    sqft_lot = Column(Float)
    floors = Column(Float)
    waterfront = Column(Integer)
    view = Column(Integer)
    condition = Column(Integer)
    grade = Column(Integer)
    yr_built = Column(Integer)
    yr_renovated = Column(Integer)
    zipcode = Column(String)
    lat = Column(Float)
    long = Column(Float)
    sqft_basement = Column(Float)

    # Model outputs
    estimated_price = Column(Float)
    status = Column(String)  # "approved" or "review"


# Physically creates the table(s) in the database, if they don't exist yet.
Base.metadata.create_all(bind=engine)
