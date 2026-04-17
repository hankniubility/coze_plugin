from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any


OBJECTIVE_KEYS = (
    "cost",
    "experience_mismatch",
    "physical_load",
    "time_inefficiency",
    "content_lack",
    "crowd_discomfort",
    "weather_mismatch",
    "continuity_detour",
)


@dataclass
class POI:
    poi_id: str
    name: str
    category: str
    tags: list[str]
    cost: float
    visit_minutes: int
    physical_intensity: float
    stairs: int
    indoor: bool
    crowd_by_hour: dict[int, float] = field(default_factory=dict)


@dataclass
class ProblemData:
    pois: list[POI]
    travel_minutes: dict[str, dict[str, float]]
    budget: float
    preferred_tags: list[str]
    day_start_hour: int
    day_end_hour: int
    target_stops: int
    weather_by_hour: dict[int, float]


@dataclass
class Individual:
    route: list[int]
    objectives: dict[str, float]
    metrics: dict[str, float]
    constraint_violation: float
    rank: int = 0
    crowding_distance: float = 0.0


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return fallback
    return numerator / denominator


def _travel_time(data: ProblemData, from_poi_id: str, to_poi_id: str) -> float:
    if from_poi_id == to_poi_id:
        return 0.0
    direct = data.travel_minutes.get(from_poi_id, {}).get(to_poi_id)
    if direct is not None:
        return float(direct)
    reverse = data.travel_minutes.get(to_poi_id, {}).get(from_poi_id)
    if reverse is not None:
        return float(reverse)
    return 35.0


def _tag_match_score(tags: list[str], preferred_tags: list[str]) -> float:
    if not preferred_tags:
        return 0.5
    tags_set = {item.lower() for item in tags}
    pref_set = {item.lower() for item in preferred_tags}
    if not tags_set:
        return 0.0
    overlap = len(tags_set & pref_set)
    return _safe_div(overlap, len(pref_set), fallback=0.0)


def _crowd_score(poi: POI, hour: int) -> float:
    if not poi.crowd_by_hour:
        return 0.5
    hour_key = int(hour) % 24
    if hour_key in poi.crowd_by_hour:
        return max(0.0, min(1.0, float(poi.crowd_by_hour[hour_key])))
    return 0.5


def _weather_penalty(data: ProblemData, poi: POI, hour: int) -> float:
    if poi.indoor:
        return 0.0
    value = data.weather_by_hour.get(int(hour) % 24, 0.15)
    return max(0.0, min(1.0, float(value)))


def _route_baseline_travel(data: ProblemData, route: list[int]) -> float:
    if len(route) <= 1:
        return 0.0

    poi_ids = [data.pois[item].poi_id for item in route]
    current = poi_ids[0]
    remaining = set(poi_ids[1:])
    baseline = 0.0
    while remaining:
        next_poi = min(remaining, key=lambda target: _travel_time(data, current, target))
        baseline += _travel_time(data, current, next_poi)
        remaining.remove(next_poi)
        current = next_poi
    return baseline


def _evaluate_route(route: list[int], data: ProblemData) -> Individual:
    pois = data.pois
    preferred_tags = data.preferred_tags

    total_cost = 0.0
    total_travel_minutes = 0.0
    total_visit_minutes = 0.0
    total_physical = 0.0
    total_match = 0.0
    total_crowd = 0.0
    total_weather = 0.0
    categories: set[str] = set()

    elapsed_minutes = 0.0
    previous: POI | None = None
    for poi_index in route:
        poi = pois[poi_index]
        travel_minutes = 0.0
        if previous is not None:
            travel_minutes = _travel_time(data, previous.poi_id, poi.poi_id)
            total_travel_minutes += travel_minutes
            elapsed_minutes += travel_minutes

        current_hour = int((data.day_start_hour * 60 + elapsed_minutes) // 60) % 24

        total_cost += poi.cost
        total_visit_minutes += poi.visit_minutes
        total_match += _tag_match_score(poi.tags, preferred_tags)
        total_crowd += _crowd_score(poi, current_hour)
        total_weather += _weather_penalty(data, poi, current_hour)
        total_physical += (
            poi.physical_intensity * (poi.visit_minutes / 60.0)
            + poi.stairs / 220.0
            + travel_minutes / 95.0
        )
        categories.add(poi.category.lower())

        elapsed_minutes += poi.visit_minutes
        previous = poi

    num_stops = len(route)
    pref_match = _safe_div(total_match, num_stops, fallback=0.0)
    crowd_avg = _safe_div(total_crowd, num_stops, fallback=0.5)
    weather_avg = _safe_div(total_weather, num_stops, fallback=0.1)
    diversity = _safe_div(len(categories), num_stops, fallback=0.0)

    total_duration = total_travel_minutes + total_visit_minutes
    available_minutes = max(60.0, (data.day_end_hour - data.day_start_hour) * 60.0)
    budget = max(1.0, data.budget)

    budget_violation = max(0.0, total_cost - budget)
    time_violation = max(0.0, total_duration - available_minutes)
    constraint_violation = _safe_div(budget_violation, budget, 0.0) + _safe_div(
        time_violation, available_minutes, 0.0
    )

    baseline_travel = _route_baseline_travel(data, route)
    detour_factor = _safe_div(total_travel_minutes, max(1.0, baseline_travel), fallback=1.0)

    objectives = {
        "cost": _safe_div(total_cost, budget, fallback=0.0),
        "experience_mismatch": max(0.0, 1.0 - pref_match),
        "physical_load": _safe_div(total_physical, num_stops, fallback=0.0),
        "time_inefficiency": _safe_div(total_travel_minutes, max(total_duration, 1.0), fallback=0.0)
        + _safe_div(time_violation, available_minutes, fallback=0.0),
        "content_lack": max(0.0, 1.0 - diversity),
        "crowd_discomfort": crowd_avg,
        "weather_mismatch": weather_avg,
        "continuity_detour": max(0.0, detour_factor - 1.0),
    }

    metrics = {
        "total_cost": round(total_cost, 3),
        "total_travel_minutes": round(total_travel_minutes, 3),
        "total_visit_minutes": round(total_visit_minutes, 3),
        "total_duration_minutes": round(total_duration, 3),
        "preference_match": round(pref_match, 4),
        "diversity_score": round(diversity, 4),
        "crowd_comfort": round(1.0 - crowd_avg, 4),
        "weather_adaptability": round(1.0 - weather_avg, 4),
        "continuity_score": round(1.0 / max(1.0, detour_factor), 4),
        "feasible": constraint_violation <= 1e-12,
        "constraint_violation": round(constraint_violation, 6),
    }

    return Individual(
        route=route,
        objectives=objectives,
        metrics=metrics,
        constraint_violation=constraint_violation,
    )


def _dominates(left: Individual, right: Individual) -> bool:
    left_cv = left.constraint_violation
    right_cv = right.constraint_violation
    if left_cv <= 1e-12 and right_cv > 1e-12:
        return True
    if left_cv > 1e-12 and right_cv <= 1e-12:
        return False
    if left_cv > 1e-12 and right_cv > 1e-12:
        return left_cv < right_cv

    better_or_equal = True
    strictly_better = False
    for key in OBJECTIVE_KEYS:
        left_value = left.objectives[key]
        right_value = right.objectives[key]
        if left_value > right_value + 1e-12:
            better_or_equal = False
            break
        if left_value < right_value - 1e-12:
            strictly_better = True
    return better_or_equal and strictly_better


def _non_dominated_sort(population: list[Individual]) -> list[list[int]]:
    domination_count = [0 for _ in population]
    dominated_set: list[list[int]] = [[] for _ in population]
    fronts: list[list[int]] = [[]]

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if _dominates(p, q):
                dominated_set[i].append(j)
            elif _dominates(q, p):
                domination_count[i] += 1
        if domination_count[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for item in fronts[front_index]:
            for dominated in dominated_set[item]:
                domination_count[dominated] -= 1
                if domination_count[dominated] == 0:
                    population[dominated].rank = front_index + 2
                    next_front.append(dominated)
        front_index += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def _assign_crowding_distance(population: list[Individual], front: list[int]) -> None:
    if not front:
        return
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = math.inf
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    for key in OBJECTIVE_KEYS:
        sorted_idx = sorted(front, key=lambda item: population[item].objectives[key])
        population[sorted_idx[0]].crowding_distance = math.inf
        population[sorted_idx[-1]].crowding_distance = math.inf

        min_value = population[sorted_idx[0]].objectives[key]
        max_value = population[sorted_idx[-1]].objectives[key]
        span = max(max_value - min_value, 1e-12)

        for pos in range(1, len(sorted_idx) - 1):
            current = sorted_idx[pos]
            if math.isinf(population[current].crowding_distance):
                continue
            prev_value = population[sorted_idx[pos - 1]].objectives[key]
            next_value = population[sorted_idx[pos + 1]].objectives[key]
            population[current].crowding_distance += (next_value - prev_value) / span


def _binary_tournament(population: list[Individual], rng: random.Random) -> Individual:
    i, j = rng.randrange(len(population)), rng.randrange(len(population))
    a, b = population[i], population[j]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b
    return a if rng.random() < 0.5 else b


def _ordered_crossover(parent_a: list[int], parent_b: list[int], rng: random.Random) -> list[int]:
    size = len(parent_a)
    if size <= 2:
        return parent_a[:]
    left, right = sorted(rng.sample(range(size), 2))
    child = [-1 for _ in range(size)]
    child[left : right + 1] = parent_a[left : right + 1]
    fill_values = [item for item in parent_b if item not in child]
    pointer = 0
    for idx in range(size):
        if child[idx] == -1:
            child[idx] = fill_values[pointer]
            pointer += 1
    return child


def _mutate(route: list[int], universe: list[int], rng: random.Random) -> list[int]:
    mutated = route[:]
    size = len(mutated)
    if size >= 2 and rng.random() < 0.7:
        i, j = rng.sample(range(size), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    if rng.random() < 0.35:
        replace_at = rng.randrange(size)
        candidate_pool = [item for item in universe if item not in mutated]
        if candidate_pool:
            mutated[replace_at] = rng.choice(candidate_pool)
    return mutated


def _initialize_population(
    poi_count: int,
    target_stops: int,
    population_size: int,
    rng: random.Random,
) -> list[list[int]]:
    universe = list(range(poi_count))
    population: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = population_size * 25
    attempts = 0
    while len(population) < population_size and attempts < max_attempts:
        route = rng.sample(universe, target_stops)
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            population.append(route)
        attempts += 1
    while len(population) < population_size:
        population.append(rng.sample(universe, target_stops))
    return population


def _weighted_score(individual: Individual, mins: dict[str, float], maxs: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(max(0.0, value) for value in weights.values())
    if total_weight <= 1e-12:
        total_weight = float(len(OBJECTIVE_KEYS))
        weights = {key: 1.0 for key in OBJECTIVE_KEYS}

    score = 0.0
    for key in OBJECTIVE_KEYS:
        value = individual.objectives[key]
        span = max(maxs[key] - mins[key], 1e-12)
        normalized = (value - mins[key]) / span
        score += normalized * max(0.0, weights.get(key, 1.0))
    return score / total_weight


def run_preference_guided_nsga2(
    data: ProblemData,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    random_seed: int | None,
    top_k: int,
    objective_weights: dict[str, float],
) -> dict[str, Any]:
    rng = random.Random(random_seed)

    poi_count = len(data.pois)
    target_stops = max(2, min(data.target_stops, poi_count))
    if poi_count < 3:
        raise ValueError("At least 3 POIs are required.")

    population_routes = _initialize_population(
        poi_count=poi_count,
        target_stops=target_stops,
        population_size=population_size,
        rng=rng,
    )
    population = [_evaluate_route(route, data) for route in population_routes]

    universe = list(range(poi_count))
    for _ in range(generations):
        fronts = _non_dominated_sort(population)
        for front in fronts:
            _assign_crowding_distance(population, front)

        offspring_routes: list[list[int]] = []
        while len(offspring_routes) < population_size:
            parent_a = _binary_tournament(population, rng)
            parent_b = _binary_tournament(population, rng)
            if rng.random() < crossover_rate:
                child = _ordered_crossover(parent_a.route, parent_b.route, rng)
            else:
                child = parent_a.route[:]

            if rng.random() < mutation_rate:
                child = _mutate(child, universe, rng)
            offspring_routes.append(child)

        offspring = [_evaluate_route(route, data) for route in offspring_routes]
        combined = population + offspring

        combined_fronts = _non_dominated_sort(combined)
        for front in combined_fronts:
            _assign_crowding_distance(combined, front)

        next_population: list[Individual] = []
        for front in combined_fronts:
            if len(next_population) + len(front) <= population_size:
                next_population.extend(combined[idx] for idx in front)
            else:
                remaining = population_size - len(next_population)
                sorted_front = sorted(
                    (combined[idx] for idx in front),
                    key=lambda item: item.crowding_distance,
                    reverse=True,
                )
                next_population.extend(sorted_front[:remaining])
                break
        population = next_population

    final_fronts = _non_dominated_sort(population)
    for front in final_fronts:
        _assign_crowding_distance(population, front)

    pareto_front = [population[idx] for idx in final_fronts[0]] if final_fronts else []
    if not pareto_front:
        pareto_front = population[:]

    mins = {key: min(item.objectives[key] for item in pareto_front) for key in OBJECTIVE_KEYS}
    maxs = {key: max(item.objectives[key] for item in pareto_front) for key in OBJECTIVE_KEYS}

    weighted_items: list[tuple[float, Individual]] = []
    for item in pareto_front:
        score = _weighted_score(item, mins, maxs, objective_weights)
        weighted_items.append((score, item))
    weighted_items.sort(key=lambda value: value[0])

    def to_plan(score_and_item: tuple[float, Individual]) -> dict[str, Any]:
        score, individual = score_and_item
        route_items = [
            {
                "poi_id": data.pois[index].poi_id,
                "name": data.pois[index].name,
                "category": data.pois[index].category,
            }
            for index in individual.route
        ]
        return {
            "route": route_items,
            "objective_vector": {key: round(individual.objectives[key], 6) for key in OBJECTIVE_KEYS},
            "metrics": individual.metrics,
            "weighted_score": round(score, 6),
            "rank": individual.rank,
            "crowding_distance": None if math.isinf(individual.crowding_distance) else round(individual.crowding_distance, 6),
        }

    plans = [to_plan(item) for item in weighted_items[: max(1, top_k)]]
    recommended = plans[0]

    feasible_count = sum(1 for item in pareto_front if item.constraint_violation <= 1e-12)
    return {
        "algorithm": {
            "name": "Preference-Guided NSGA-II",
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "random_seed": random_seed,
        },
        "summary": {
            "candidate_poi_count": poi_count,
            "target_stops": target_stops,
            "pareto_count": len(pareto_front),
            "feasible_pareto_count": feasible_count,
            "returned_plan_count": len(plans),
        },
        "objective_definition": {
            "cost": "Lower is better. Cost/Budget ratio with over-budget penalty reflected by constraints.",
            "experience_mismatch": "Lower is better. 1 - preference_match_score.",
            "physical_load": "Lower is better. Composite of activity intensity, stairs, and movement effort.",
            "time_inefficiency": "Lower is better. Travel share plus overtime penalty.",
            "content_lack": "Lower is better. 1 - category diversity score.",
            "crowd_discomfort": "Lower is better. Estimated crowd index along the schedule.",
            "weather_mismatch": "Lower is better. Outdoor exposure under bad weather windows.",
            "continuity_detour": "Lower is better. Excess detour factor versus a greedy baseline route.",
        },
        "recommended_plan": recommended,
        "pareto_plans": plans,
    }

