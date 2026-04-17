from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .pareto_optimizer import OBJECTIVE_KEYS, POI, ProblemData, run_preference_guided_nsga2


class ParetoPOIInput(BaseModel):
    poi_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    category: str = Field(default="other", min_length=1, max_length=64)
    tags: list[str] = Field(default_factory=list)
    cost: float = Field(default=0.0, ge=0.0)
    visit_minutes: int = Field(default=60, ge=10, le=720)
    physical_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    stairs: int = Field(default=0, ge=0)
    indoor: bool = False
    crowd_by_hour: dict[int, float] = Field(
        default_factory=dict,
        description="Hour->crowd score mapping, each value in [0,1].",
    )


class ParetoUserPreferenceInput(BaseModel):
    budget: float = Field(..., gt=0.0)
    preferred_tags: list[str] = Field(default_factory=list)
    day_start_hour: int = Field(default=9, ge=0, le=23)
    day_end_hour: int = Field(default=21, ge=1, le=24)
    target_stops: int = Field(default=5, ge=2, le=20)


class ParetoWeightsInput(BaseModel):
    cost: float = Field(default=1.0, ge=0.0)
    experience: float = Field(default=1.0, ge=0.0)
    physical: float = Field(default=1.0, ge=0.0)
    time_efficiency: float = Field(default=1.0, ge=0.0)
    content_richness: float = Field(default=1.0, ge=0.0)
    crowd_comfort: float = Field(default=1.0, ge=0.0)
    weather_adaptability: float = Field(default=1.0, ge=0.0)
    continuity: float = Field(default=1.0, ge=0.0)

    def to_objective_weights(self) -> dict[str, float]:
        return {
            "cost": self.cost,
            "experience_mismatch": self.experience,
            "physical_load": self.physical,
            "time_inefficiency": self.time_efficiency,
            "content_lack": self.content_richness,
            "crowd_discomfort": self.crowd_comfort,
            "weather_mismatch": self.weather_adaptability,
            "continuity_detour": self.continuity,
        }


class ParetoOptimizeRequest(BaseModel):
    pois: list[ParetoPOIInput] = Field(..., min_length=3, max_length=120)
    travel_minutes: dict[str, dict[str, float]] = Field(
        ...,
        description="Travel matrix in minutes. Missing edges will use fallback travel time.",
    )
    user_preference: ParetoUserPreferenceInput
    weather_by_hour: dict[int, float] = Field(
        default_factory=dict,
        description="Hour->outdoor risk in [0,1], e.g. rain/high-heat windows.",
    )
    weights: ParetoWeightsInput = Field(default_factory=ParetoWeightsInput)
    population_size: int = Field(default=80, ge=20, le=300)
    generations: int = Field(default=100, ge=10, le=1000)
    crossover_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    random_seed: int | None = None
    top_k: int = Field(default=15, ge=1, le=100)


router = APIRouter(prefix="/api/plugin/pareto", tags=["Pareto Optimization Plugin"])


@router.get("/health")
def pareto_health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/optimize")
def pareto_optimize(payload: ParetoOptimizeRequest) -> dict:
    if payload.user_preference.day_end_hour <= payload.user_preference.day_start_hour:
        raise HTTPException(status_code=400, detail="day_end_hour must be greater than day_start_hour.")

    if payload.user_preference.target_stops > len(payload.pois):
        raise HTTPException(status_code=400, detail="target_stops cannot exceed the number of POIs.")

    pois = [
        POI(
            poi_id=item.poi_id,
            name=item.name,
            category=item.category,
            tags=item.tags,
            cost=item.cost,
            visit_minutes=item.visit_minutes,
            physical_intensity=item.physical_intensity,
            stairs=item.stairs,
            indoor=item.indoor,
            crowd_by_hour={int(hour): float(value) for hour, value in item.crowd_by_hour.items()},
        )
        for item in payload.pois
    ]

    for row_key, row_values in payload.travel_minutes.items():
        for col_key, value in row_values.items():
            if value < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"travel_minutes[{row_key}][{col_key}] must be >= 0.",
                )

    for hour, value in payload.weather_by_hour.items():
        if hour < 0 or hour > 23:
            raise HTTPException(status_code=400, detail="weather_by_hour hour key must be between 0 and 23.")
        if value < 0 or value > 1:
            raise HTTPException(status_code=400, detail="weather_by_hour values must be in [0,1].")

    data = ProblemData(
        pois=pois,
        travel_minutes=payload.travel_minutes,
        budget=payload.user_preference.budget,
        preferred_tags=payload.user_preference.preferred_tags,
        day_start_hour=payload.user_preference.day_start_hour,
        day_end_hour=payload.user_preference.day_end_hour,
        target_stops=payload.user_preference.target_stops,
        weather_by_hour={int(hour): float(value) for hour, value in payload.weather_by_hour.items()},
    )

    objective_weights = payload.weights.to_objective_weights()
    if set(objective_weights.keys()) != set(OBJECTIVE_KEYS):
        raise HTTPException(status_code=400, detail="Invalid objective weight mapping.")

    try:
        result = run_preference_guided_nsga2(
            data=data,
            population_size=payload.population_size,
            generations=payload.generations,
            crossover_rate=payload.crossover_rate,
            mutation_rate=payload.mutation_rate,
            random_seed=payload.random_seed,
            top_k=payload.top_k,
            objective_weights=objective_weights,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result

