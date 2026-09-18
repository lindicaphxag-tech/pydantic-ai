from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import httpx2

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from bench_union import CASES, Escalation, Ticket, _run_all


async def run(output_type: object) -> tuple[list[object], dict[str, list[bytes]]]:
    bodies: dict[str, list[bytes]] = defaultdict(list)

    async def capture(request: httpx2.Request) -> None:
        content = bytes(request.content)
        state = json.loads(content)['state']
        assert isinstance(state, str)
        bodies[state].append(content)

    async with httpx2.AsyncClient(event_hooks={'request': [capture]}) as client:
        model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(http_client=client))
        rows, _ = await _run_all(lambda: Agent(model, output_type=output_type), concurrency=8)
    return rows, bodies


async def main() -> None:
    baseline, baseline_bodies = await run(Ticket)
    union, union_bodies = await run(Ticket | Escalation)

    ticket_count = escalation_count = 0
    ticket_urgent_flips = escalation_urgent_flips = 0
    ticket_area_flips = escalation_area_flips = 0
    ticket_wire_equal = 0
    ticket_question_equal = 0

    for baseline_row, union_row, (text, _, _) in zip(baseline, union, CASES):
        assert baseline_row.error is None and union_row.error is None
        assert len(baseline_bodies[text]) == 1
        assert len(union_bodies[text]) == 2
        baseline_body = baseline_bodies[text][0]
        second_body = next(
            body for body in union_bodies[text] if 'urgent' in json.loads(body)['questions']
        )
        if union_row.escalation:
            escalation_count += 1
            escalation_urgent_flips += baseline_row.urgent != union_row.urgent
            escalation_area_flips += baseline_row.area != union_row.area
        else:
            ticket_count += 1
            ticket_urgent_flips += baseline_row.urgent != union_row.urgent
            ticket_area_flips += baseline_row.area != union_row.area
            ticket_wire_equal += baseline_body == second_body
            ticket_question_equal += json.loads(baseline_body)['questions'] == json.loads(second_body)['questions']

    print(f'chosen Ticket: {ticket_count}')
    print(f'chosen Escalation: {escalation_count}')
    print(f'urgent flips for Ticket picks: {ticket_urgent_flips}')
    print(f'urgent flips for Escalation picks: {escalation_urgent_flips}')
    print(f'area flips for Ticket picks: {ticket_area_flips}')
    print(f'area flips for Escalation picks: {escalation_area_flips}')
    print(f'Ticket second body exact matches: {ticket_wire_equal}/{ticket_count}')
    print(f'Ticket second questions matches: {ticket_question_equal}/{ticket_count}')


if __name__ == '__main__':
    asyncio.run(main())
