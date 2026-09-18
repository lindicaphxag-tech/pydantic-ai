"""Measure Jev's choose-then-fill union against one-call and frontier-model baselines.

The 120 cases and their labels come from ``cases.py``. ``Escalation`` is right when the
rule says the ticket is urgent and owned by account or bug; billing urgency stays a ticket.
"""

from __future__ import annotations

import asyncio
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import Agent, NativeOutput

os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from cases import CASES, Area

N = len(CASES)


class Ticket(BaseModel):
    """Keep this support ticket in the normal queue rather than handing it to a person now."""

    model_config = ConfigDict(extra='forbid')

    urgent: bool = Field(description='Does this need a reply within the hour?')
    area: Area = Field(description='Which team owns it?')


class Escalation(BaseModel):
    """Hand this to a person now: the customer is blocked, the product is broadly down, or access is at risk."""

    model_config = ConfigDict(extra='forbid')

    urgent: bool = Field(description='Does this need a reply within the hour?')
    area: Area = Field(description='Which team owns it?')
    reason: Literal['blocked', 'outage', 'access_risk'] = Field(description='Why does a person need to take it now?')


class Flattened(BaseModel):
    """Choose whether to keep a support ticket in the normal queue or hand it to a person now."""

    kind: Literal['ticket', 'escalation'] = Field(description='What should happen to this ticket?')
    urgent: Literal['yes', 'no'] | None = Field(description='Does this need a reply within the hour?')
    area: Area | None = Field(description='Which team owns it?')
    reason: Literal['blocked', 'outage', 'access_risk'] | None = Field(
        description='Why does a person need to take it now, if it does?'
    )


@dataclass
class _Row:
    latency: float
    escalation: bool | None
    urgent: bool | None
    area: Area | None
    cost: Decimal | None
    error: str | None


def _wants_escalation(urgent: bool, area: Area) -> bool:
    return urgent and area in ('account', 'bug')


def _cost_of(result: object) -> Decimal | None:
    value = getattr(getattr(result, 'usage'), 'cost', None)
    return value if isinstance(value, Decimal) else None


async def _run_all(make_agent: Callable[[], Agent[None, object]], *, concurrency: int) -> tuple[list[_Row], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(text: str) -> _Row:
        async with semaphore:
            started = time.perf_counter()
            try:
                result = await make_agent().run(text)
                output = result.output
                if isinstance(output, Flattened):
                    escalation = output.kind == 'escalation'
                    urgent = None if output.urgent is None else output.urgent == 'yes'
                    area = output.area
                else:
                    escalation = isinstance(output, Escalation)
                    urgent = output.urgent
                    area = output.area
                return _Row(time.perf_counter() - started, escalation, urgent, area, _cost_of(result), None)
            except Exception as error:
                return _Row(
                    time.perf_counter() - started,
                    None,
                    None,
                    None,
                    None,
                    f'{type(error).__name__}: {str(error)[:120]}',
                )

    wall_started = time.perf_counter()
    rows = await asyncio.gather(*(one(text) for text, _, _ in CASES))
    return rows, time.perf_counter() - wall_started


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _routing_summary(label: str, rows: list[_Row], wall: float) -> str:
    latencies = [row.latency for row in rows]
    right = sum(
        row.escalation == _wants_escalation(urgent, area)
        for row, (_, urgent, area) in zip(rows, CASES)
        if row.escalation is not None
    )
    hand_offs = sum(row.escalation is True for row in rows)
    errors = sum(row.error is not None for row in rows)
    costs = [row.cost for row in rows if row.cost is not None]
    cost = f'${sum(costs, Decimal()):.4f}' if len(costs) == len(rows) else 'unavailable'
    return (
        f'| {label} | {right}/{N} ({right / N:.1%}) | {hand_offs}/{N} ({hand_offs / N:.1%}) | '
        f'{statistics.median(latencies) * 1000:.0f} ms | {_percentile(latencies, 0.95) * 1000:.0f} ms | '
        f'{wall:.1f} s | {cost} | {errors} |'
    )


def _field_flips(left: list[_Row], right: list[_Row]) -> tuple[int, int, int]:
    urgent = sum(a.urgent != b.urgent for a, b in zip(left, right))
    area = sum(a.area != b.area for a, b in zip(left, right))
    either = sum((a.urgent, a.area) != (b.urgent, b.area) for a, b in zip(left, right))
    return urgent, area, either


def _error_notes(named_rows: list[tuple[str, list[_Row]]]) -> list[str]:
    notes: list[str] = []
    for label, rows in named_rows:
        errors: dict[str, int] = {}
        for row in rows:
            if row.error:
                errors[row.error] = errors.get(row.error, 0) + 1
        if errors:
            notes.append(f'- {label}: ' + '; '.join(f'{count} × {error}' for error, count in errors.items()))
    return notes


async def _main() -> None:
    def ticket() -> Agent[None, Ticket]:
        return Agent('typesafe:jev-latest', output_type=Ticket)

    def union() -> Agent[None, Ticket | Escalation]:
        return Agent('typesafe:jev-latest', output_type=Ticket | Escalation)

    def flattened() -> Agent[None, Flattened]:
        return Agent('typesafe:jev-latest', output_type=Flattened)

    baseline_1, baseline_1_wall = await _run_all(ticket, concurrency=8)
    baseline_2, baseline_2_wall = await _run_all(ticket, concurrency=8)
    jev_union, jev_union_wall = await _run_all(union, concurrency=8)
    flat, flat_wall = await _run_all(flattened, concurrency=8)

    native_results: list[tuple[str, list[_Row], float]] = []
    for label, model in [
        ('Sol NativeOutput union', 'openai:gpt-5.6-sol'),
        ('Opus 5 NativeOutput union', 'anthropic:claude-opus-5'),
    ]:
        rows, wall = await _run_all(
            lambda model=model: Agent(model, output_type=NativeOutput(Ticket | Escalation), retries=2),
            concurrency=6,
        )
        native_results.append((label, rows, wall))

    pass_flips = _field_flips(baseline_1, baseline_2)
    union_flips = _field_flips(baseline_1, jev_union)
    lines = [
        '# Union benchmark — 120 rule-labelled support tickets',
        '',
        'Rule: escalate exactly urgent account or bug tickets (32/120); urgent billing remains a ticket.',
        '',
        '## Contamination',
        '',
        '| comparison | urgent flips | area flips | either field flips | wall time |',
        '|---|---:|---:|---:|---:|',
        f'| Ticket pass 1 vs Ticket pass 2 (noise floor) | {pass_flips[0]} | {pass_flips[1]} | {pass_flips[2]} | '
        f'{baseline_1_wall + baseline_2_wall:.1f} s |',
        f'| Ticket pass 1 vs Ticket \\| Escalation | {union_flips[0]} | {union_flips[1]} | {union_flips[2]} | '
        f'{jev_union_wall:.1f} s |',
        '',
        'Both union members carry the same `urgent` and `area` fields, so all 120 answers are compared. '
        '`Escalation` adds only its reason.',
        '',
        '## Union routing',
        '',
        '| model | accuracy | hand-off rate | median | p95 | wall | cost | errors |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        _routing_summary('Jev choose then fill', jev_union, jev_union_wall),
        *(_routing_summary(label, rows, wall) for label, rows, wall in native_results),
        '',
        '## Two calls vs one flattened call',
        '',
        '| Jev shape | routing accuracy | escalation rate | median | p95 | wall | cost | errors |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        _routing_summary('Two-call Ticket \\| Escalation', jev_union, jev_union_wall),
        _routing_summary('One-call flattened discriminator', flat, flat_wall),
        '',
        'The flattened shape has `kind` plus optional fields for the two members; the two-call shape asks only the '
        'chosen member.',
    ]
    errors = _error_notes(
        [
            ('Ticket pass 1', baseline_1),
            ('Ticket pass 2', baseline_2),
            ('Jev union', jev_union),
            ('Jev flattened', flat),
            *((label, rows) for label, rows, _ in native_results),
        ]
    )
    if errors:
        lines.extend(['', '## Errors', '', *errors])
    print('\n'.join(lines))


if __name__ == '__main__':
    asyncio.run(_main())
