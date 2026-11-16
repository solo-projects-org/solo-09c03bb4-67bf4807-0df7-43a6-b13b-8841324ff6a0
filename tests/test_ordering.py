"""Tests for the batch-ordering API (POST /v1/orders)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import AuthoritativeSnapshot, Batch, InitialData, PrecedenceConstraint
from app.ordering import (
    STATUS_AMBIGUOUS,
    STATUS_NO_SOLUTION,
    STATUS_SEARCH_LIMIT,
    STATUS_UNIQUE,
    OrderSearcher,
)
from app.validate import InputError

from conftest import op_delete, op_insert, op_update

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _client() -> TestClient:
    return TestClient(app)


def _initial() -> InitialData:
    return InitialData.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )


def _order_batches() -> list[Batch]:
    """Batches submitted out of order: 3, 1, 4, 2.

    Forced chain: 1 inserts order 3; 2 renames order 3 -> 30; 3 updates the
    amount of order 30 (target only exists after 2); 4 renames 30 -> 300 and
    sets the final amount (target only exists after 3).  Each later batch is
    illegal before its predecessor, so the order is unique with no precedence
    constraints at all.
    """
    return [
        Batch.model_validate(
            {
                "batch_id": 3,
                "changes": [
                    op_update(
                        "orders", {"order_id": 30}, {"order_id": 30, "amount": 800}
                    )
                ],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 1,
                "changes": [op_insert("orders", order_id=3, amount=700)],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 4,
                "changes": [
                    op_update(
                        "orders", {"order_id": 30}, {"order_id": 300, "amount": 900}
                    )
                ],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 2,
                "changes": [
                    op_update(
                        "orders", {"order_id": 3}, {"order_id": 30, "amount": 700}
                    )
                ],
            }
        ),
    ]


def _authoritative_after_correct_order() -> AuthoritativeSnapshot:
    return AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 300, "amount": 900},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )


def _precedence(*pairs: tuple[int, int]) -> list[PrecedenceConstraint]:
    return [
        PrecedenceConstraint(before_batch_id=before, after_batch_id=after)
        for before, after in pairs
    ]


def _search(
    batches: list[Batch],
    authoritative: AuthoritativeSnapshot | None = None,
    precedence: list[PrecedenceConstraint] | None = None,
    **kwargs,
):
    return OrderSearcher(
        _initial(),
        batches,
        precedence or [],
        authoritative or _authoritative_after_correct_order(),
        **kwargs,
    ).run()


# ---------------------------------------------------------------------------
# Core search semantics
# ---------------------------------------------------------------------------


def test_unique_order_is_recovered_from_out_of_order_input():
    outcome = _search(_order_batches())
    assert outcome.status == STATUS_UNIQUE
    assert outcome.orderings == [[1, 2, 3, 4]]
    assert outcome.search_complete is True


def test_unique_order_uses_every_batch_and_satisfies_precedence():
    # Redundant precedence edges (including a transitive one) must not change
    # the unique answer.
    outcome = _search(
        _order_batches(),
        precedence=_precedence((1, 2), (2, 3), (1, 3), (1, 3)),
    )
    assert outcome.status == STATUS_UNIQUE
    assert outcome.orderings == [[1, 2, 3, 4]]


def test_two_independent_orders_are_ambiguous():
    # Batch 5 can run anywhere before batch 1 renames order 3; pairing it with
    # the unique chain yields two distinct feasible orders.
    batches = _order_batches() + [
        Batch.model_validate(
            {
                "batch_id": 5,
                "changes": [op_insert("orders", order_id=4, amount=400)],
            }
        ),
    ]
    authoritative = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 4, "amount": 400},
                {"order_id": 300, "amount": 900},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )
    outcome = _search(batches, authoritative)
    assert outcome.status == STATUS_AMBIGUOUS
    assert len(outcome.orderings) == 2
    assert outcome.orderings[0] != outcome.orderings[1]
    assert outcome.search_complete is False
    for order in outcome.orderings:
        assert sorted(order) == [1, 2, 3, 4, 5]


def test_no_solution_when_authoritative_snapshot_never_reached():
    wrong = AuthoritativeSnapshot.model_validate(
        {
            "orders": [{"order_id": 1, "amount": 1000}],
            "refunds": [],
        }
    )
    outcome = _search(_order_batches(), wrong)
    assert outcome.status == STATUS_NO_SOLUTION
    assert outcome.orderings == []
    assert outcome.search_complete is True


def test_no_solution_when_precedence_constraints_are_cyclic():
    outcome = _search(
        _order_batches(), precedence=_precedence((1, 2), (2, 1))
    )
    assert outcome.status == STATUS_NO_SOLUTION
    assert outcome.search_complete is True
    assert any("环" in note for note in outcome.notes)


def test_precedence_can_force_a_different_unique_order():
    # Batch 2 renames order 3 -> 30, batch 1 inserts order 3; requiring 2
    # before 1 makes batch 2 illegal first, so no order can satisfy both the
    # relation and the snapshot.
    outcome = _search(_order_batches(), precedence=_precedence((2, 1)))
    assert outcome.status == STATUS_NO_SOLUTION


def test_authoritative_snapshot_row_order_is_irrelevant():
    shuffled = AuthoritativeSnapshot.model_validate(
        {
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
            "orders": [
                {"order_id": 300, "amount": 900},
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
        }
    )
    assert _search(_order_batches(), shuffled).status == STATUS_UNIQUE


def test_intra_batch_atomicity_permits_deferred_foreign_keys():
    # One batch moves the order key and the referencing refund together; it
    # must be accepted (same rules as the executor).
    batches = [
        Batch.model_validate(
            {
                "batch_id": 1,
                "changes": [
                    op_update(
                        "orders", {"order_id": 1}, {"order_id": 101, "amount": 1000}
                    ),
                    op_update(
                        "refunds",
                        {"refund_id": 10},
                        {"refund_id": 10, "order_id": 101, "amount": 200},
                    ),
                ],
            }
        )
    ]
    authoritative = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 101, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
            "refunds": [{"refund_id": 10, "order_id": 101, "amount": 200}],
        }
    )
    outcome = _search(batches, authoritative)
    assert outcome.status == STATUS_UNIQUE
    assert outcome.orderings == [[1]]


def test_batch_that_leaves_dangling_foreign_key_is_not_feasible():
    batches = [
        Batch.model_validate(
            {
                "batch_id": 1,
                "changes": [op_delete("refunds", refund_id=10)],
            }
        ),
        Batch.model_validate(
            {"batch_id": 2, "changes": [op_delete("orders", order_id=1)]},
        ),
    ]
    # Batch 2 can never be applied while refund 10 references order 1; if
    # batch 1 is placed first, the final state is not the snapshot.
    authoritative = AuthoritativeSnapshot.model_validate(
        {
            "orders": [{"order_id": 2, "amount": 500}],
            "refunds": [],
        }
    )
    outcome = _search(batches, authoritative)
    # 1 then 2 is actually feasible: deleting the refund first satisfies the
    # FK, so this is unique rather than impossible.
    assert outcome.status == STATUS_UNIQUE
    assert outcome.orderings == [[1, 2]]


def test_single_batch_permutation_is_trivially_unique():
    batches = [
        Batch.model_validate(
            {
                "batch_id": 9,
                "changes": [op_insert("orders", order_id=9, amount=90)],
            }
        )
    ]
    authoritative = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 9, "amount": 90},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )
    outcome = _search(batches, authoritative)
    assert outcome.status == STATUS_UNIQUE
    assert outcome.orderings == [[9]]


# ---------------------------------------------------------------------------
# Search budget / incomplete search
# ---------------------------------------------------------------------------


def test_node_budget_exhaustion_reports_search_limit_not_no_solution():
    outcome = _search(_order_batches(), node_budget=0)
    assert outcome.status == STATUS_SEARCH_LIMIT
    assert outcome.search_complete is False
    assert any("不能据此宣称无解" in note for note in outcome.notes)


def test_node_budget_exhaustion_with_one_solution_never_claims_unique():
    # Budget 1: enough to expand the only first step (batch 1) but not to
    # finish.  Even if a feasible order had been found, the search is
    # unfinished and must not be called unique.
    outcome = _search(_order_batches(), node_budget=1)
    assert outcome.status == STATUS_SEARCH_LIMIT
    assert outcome.search_complete is False
    assert outcome.orderings == []


def test_time_budget_exhaustion_reports_search_limit():
    outcome = _search(_order_batches(), time_budget=-1.0)
    assert outcome.status == STATUS_SEARCH_LIMIT
    assert outcome.search_complete is False


def test_budget_exhaustion_after_a_solution_does_not_claim_unique():
    # Find a solution first, then run out of nodes while exploring siblings.
    full = _search(_order_batches())
    assert full.status == STATUS_UNIQUE
    outcome = _search(_order_batches(), node_budget=full.searched_nodes - 1)
    assert outcome.status == STATUS_SEARCH_LIMIT
    assert outcome.search_complete is False


def test_searched_nodes_are_reported():
    outcome = _search(_order_batches())
    assert outcome.searched_nodes >= 4


def test_full_exhaustion_of_8_free_batches_stays_within_node_budget():
    # The worst case for "no_solution" is 8 batches with no precedence
    # constraints and an unreachable snapshot: all 109600 prefix extensions
    # must be tried, and the search must still finish (no false
    # ``search_limit``) within the default node budget.
    batches = [
        Batch.model_validate(
            {
                "batch_id": i,
                "changes": [
                    op_insert("orders", order_id=100 + i, amount=i)
                ],
            }
        )
        for i in range(1, 9)
    ]
    outcome = _search(batches)
    assert outcome.status == STATUS_NO_SOLUTION
    assert outcome.search_complete is True
    assert outcome.searched_nodes < 200_000


# ---------------------------------------------------------------------------
# Final-state semantics
# ---------------------------------------------------------------------------


def test_snapshot_value_mismatch_is_not_feasible():
    wrong_amount = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 300, "amount": 901},  # off by one
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )
    assert _search(_order_batches(), wrong_amount).status == STATUS_NO_SOLUTION


def test_snapshot_extra_row_is_not_feasible():
    with_extra = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 300, "amount": 900},
                {"order_id": 999, "amount": 1},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )
    assert _search(_order_batches(), with_extra).status == STATUS_NO_SOLUTION


def test_snapshot_missing_refund_is_not_feasible():
    with_extra_refund = AuthoritativeSnapshot.model_validate(
        {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 300, "amount": 900},
            ],
            "refunds": [],
        }
    )
    assert _search(_order_batches(), with_extra_refund).status == STATUS_NO_SOLUTION


def test_permutation_must_use_all_batches():
    # The first four batches alone reach the snapshot; any order that also
    # applies batch 88 inserts an extra order and therefore cannot match it.
    # If the search were allowed to skip batches, batch 88 could be ignored --
    # no_solution proves every submitted batch must be consumed.
    batches = _order_batches() + [
        Batch.model_validate(
            {
                "batch_id": 88,
                "changes": [op_insert("orders", order_id=999, amount=1)],
            }
        )
    ]
    outcome = _search(batches)
    assert outcome.status == STATUS_NO_SOLUTION
    assert outcome.search_complete is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unknown_precedence_batch_is_an_input_error():
    with pytest.raises(InputError, match="不是已提交的批次号"):
        _search(_order_batches(), precedence=_precedence((1, 99)))


def test_authoritative_snapshot_must_be_consistent():
    broken = AuthoritativeSnapshot.model_validate(
        {
            "orders": [{"order_id": 1, "amount": 1}],
            "refunds": [{"refund_id": 10, "order_id": 2, "amount": 1}],
        }
    )
    with pytest.raises(InputError, match="不存在的 order_id"):
        _search(_order_batches(), broken)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def _api_payload() -> dict:
    return {
        "initial": {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        },
        "batches": [b.model_dump(mode="json", exclude_none=True) for b in _order_batches()],
        "precedence": [{"before_batch_id": 1, "after_batch_id": 3}],
        "authoritative": _authoritative_after_correct_order().model_dump(mode="json"),
    }


def test_api_unique():
    response = _client().post("/v1/orders", json=_api_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unique"
    assert body["orderings"] == [[1, 2, 3, 4]]
    assert body["search_complete"] is True
    assert body["searched_nodes"] > 0
    assert body["notes"]


def test_api_ambiguous_reports_two_distinct_orders():
    payload = _api_payload()
    payload["batches"].append(
        {
            "batch_id": 5,
            "changes": [op_insert("orders", order_id=4, amount=400)],
        }
    )
    payload["authoritative"]["orders"].append({"order_id": 4, "amount": 400})
    response = _client().post("/v1/orders", json=payload)
    body = response.json()
    assert body["status"] == "ambiguous"
    assert len(body["orderings"]) == 2
    assert body["orderings"][0] != body["orderings"][1]
    assert body["search_complete"] is False


def test_api_no_solution():
    payload = _api_payload()
    payload["authoritative"]["orders"][0]["amount"] = 1
    body = _client().post("/v1/orders", json=payload).json()
    assert body["status"] == "no_solution"
    assert body["orderings"] == []
    assert body["search_complete"] is True


def test_api_more_than_8_batches_rejected():
    payload = _api_payload()
    payload["batches"] = [
        {"batch_id": i, "changes": [op_insert("orders", order_id=100 + i, amount=1)]}
        for i in range(1, 10)
    ]
    response = _client().post("/v1/orders", json=payload)
    assert response.status_code == 422


def test_api_unknown_precedence_batch_rejected():
    payload = _api_payload()
    payload["precedence"] = [{"before_batch_id": 1, "after_batch_id": 77}]
    response = _client().post("/v1/orders", json=payload)
    assert response.status_code == 422
    assert "不是已提交的批次号" in response.json()["detail"]


def test_api_self_precedence_rejected_by_model():
    payload = _api_payload()
    payload["precedence"] = [{"before_batch_id": 1, "after_batch_id": 1}]
    assert _client().post("/v1/orders", json=payload).status_code == 422


def test_api_duplicate_batch_ids_rejected():
    payload = _api_payload()
    payload["batches"][1]["batch_id"] = payload["batches"][0]["batch_id"]
    assert _client().post("/v1/orders", json=payload).status_code == 422


def test_api_missing_initial_order_reference_rejected():
    payload = _api_payload()
    payload["initial"]["refunds"] = [
        {"refund_id": 10, "order_id": 999, "amount": 200}
    ]
    assert _client().post("/v1/orders", json=payload).status_code == 422


def test_api_example_file_unique():
    payload = json.loads((EXAMPLES / "order_batches.json").read_text())
    response = _client().post("/v1/orders", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unique"
    assert body["orderings"] == [[201, 202, 203, 204, 205]]
    assert body["search_complete"] is True


# ---------------------------------------------------------------------------
# Cross-check against the real executor (SQLite batch transactions + FKs)
# ---------------------------------------------------------------------------

_RECOMPUTE_INCREMENTAL = (
    "DELETE FROM result;"
    "INSERT INTO result(order_id, net_amount) "
    "SELECT o.order_id, o.amount - COALESCE(SUM(r.amount), 0) "
    "FROM orders o LEFT JOIN refunds r ON r.order_id = o.order_id "
    "GROUP BY o.order_id, o.amount;"
)


def _net_full_select() -> str:
    return (
        "SELECT o.order_id, o.amount - COALESCE(SUM(r.amount), 0) AS net_amount "
        "FROM orders o LEFT JOIN refunds r ON r.order_id = o.order_id "
        "GROUP BY o.order_id, o.amount"
    )


def test_recovered_order_is_accepted_by_the_real_executor():
    # The recovered order must pass the *same* batch-transaction and foreign
    # key rules as the replayer.  A "recompute result from source" incremental
    # SQL makes every batch's post-commit state comparable, so outcome ``ok``
    # means all batches were legal and applied in the recovered order.
    from app.executor import Executor

    def batches_by_id() -> dict[int, Batch]:
        return {batch.batch_id: batch for batch in _order_batches()}

    by_id = batches_by_id()
    recovered = _search(_order_batches()).orderings[0]
    executor = Executor(_net_full_select(), _RECOMPUTE_INCREMENTAL)

    ordered = executor.run(_initial(), [by_id[i] for i in recovered])
    assert ordered.outcome == "ok", ordered.notes

    # The submitted (out-of-order) sequence is not executable as-is: batch 3
    # updates order 30 before batch 2 creates it.
    submitted = _order_batches()
    as_submitted = executor.run(_initial(), submitted)
    assert as_submitted.outcome == "error"


def test_recovered_order_also_works_through_the_validate_endpoint():
    payload = json.loads((EXAMPLES / "order_batches.json").read_text())
    body = _client().post("/v1/orders", json=payload).json()
    order = body["orderings"][0]
    by_id = {batch["batch_id"]: batch for batch in payload["batches"]}

    validate_payload = {
        "initial": payload["initial"],
        "batches": [by_id[i] for i in order],
        "sql": {
            "full_select": _net_full_select(),
            "incremental": _RECOMPUTE_INCREMENTAL,
        },
    }
    response = _client().post("/v1/validate", json=validate_payload)
    assert response.status_code == 200
    assert response.json()["outcome"] == "ok"


def test_existing_endpoints_are_unchanged():
    schema = _client().get("/openapi.json").json()
    assert "/v1/minimize" in schema["paths"]
    assert "/v1/validate" in schema["paths"]
    assert "/v1/orders" in schema["paths"]
