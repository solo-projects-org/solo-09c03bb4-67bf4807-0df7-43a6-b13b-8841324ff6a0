"""Recover an executable order for out-of-order CDC batches.

Unlike the minimizer endpoints, the batch order as submitted is *not*
assumed to be executable: the whole point of this API is to recover an
executable order.  A candidate order is feasible exactly when

* it is a permutation of **all** submitted batches (no batch may be skipped);
* every mandatory ``precedence`` relation is honoured (``before`` placed
  earlier than ``after``);
* every batch is legal when it is applied, using the same rules as the
  executor -- changes inside one batch are applied like one transaction with
  deferred foreign-key checks (so a batch may move an order key and the
  referencing refunds together), the update/delete target must exist, primary
  keys stay unique, and the foreign key must hold after the batch;
* the state after the last batch equals the authoritative ``orders`` /
  ``refunds`` snapshot exactly (every row and every value; row order is
  irrelevant).

The search is a depth-first enumeration over permutations that consumes one
batch at each level.  Batches whose predecessor set is not yet placed are not
candidates; a precedence cycle is detected once (Kahn's algorithm) and proves
unsolvability without expanding any branch.  Because the effect of each batch
is deterministic, a prefix that cannot extend legally is pruned immediately.

Output contract: ``unique`` and ``no_solution`` may only be reported when the
enumeration *exhausted* the search space.  The search stops as soon as two
different feasible orders are found (that already proves ``ambiguous``); any
budget/time exhaustion yields ``search_limit`` and never counts unvisited
branches as evidence for uniqueness or unsolvability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .models import (
    AuthoritativeSnapshot,
    Batch,
    InitialData,
    PrecedenceConstraint,
)
from .validate import (
    InputError,
    SourceState,
    apply_batch_recording,
    build_authoritative_state,
    build_initial_state,
    same_state,
    validate_batch_shapes,
    validate_precedence,
)

STATUS_UNIQUE = "unique"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NO_SOLUTION = "no_solution"
STATUS_SEARCH_LIMIT = "search_limit"


@dataclass
class OrderingOutcome:
    status: str
    orderings: list[list[int]] = field(default_factory=list)
    search_complete: bool = False
    searched_nodes: int = 0
    notes: list[str] = field(default_factory=list)


class OrderSearcher:
    """Depth-first search for all feasible batch orderings (up to two)."""

    def __init__(
        self,
        initial: InitialData,
        batches: list[Batch],
        precedence: list[PrecedenceConstraint],
        authoritative: AuthoritativeSnapshot,
        *,
        node_budget: int | None = None,
        time_budget: float | None = None,
    ) -> None:
        self.initial = initial
        self.batches = list(batches)
        self.precedence = list(precedence)
        self.authoritative = authoritative
        self.node_budget = (
            config.MAX_ORDER_SEARCH_NODES if node_budget is None else node_budget
        )
        self.time_budget = (
            config.MAX_ORDER_SECONDS if time_budget is None else time_budget
        )

        self.nodes = 0
        self.solutions: list[list[int]] = []
        self.aborted = False
        self.early_stopped = False
        self.cycle_detected = False
        self.pruned_invalid = 0
        self._deadline = 0.0
        self._preds: list[set[int]] = []
        self._target: SourceState | None = None

    # -- budget ----------------------------------------------------------
    def _consume_budget(self) -> bool:
        if self.nodes >= self.node_budget or time.monotonic() > self._deadline:
            self.aborted = True
            return False
        self.nodes += 1
        return True

    # -- public entry point ----------------------------------------------
    def run(self) -> OrderingOutcome:
        # Malformed inputs (as opposed to merely unsatisfiable ones) are input
        # errors: the API edge reports them as 422.
        start = build_initial_state(self.initial)
        validate_batch_shapes(self.batches)
        validate_precedence(self.batches, self.precedence)
        target = build_authoritative_state(self.authoritative)
        self._target = target

        position = {batch.batch_id: index for index, batch in enumerate(self.batches)}
        self._preds = [set() for _ in self.batches]
        for relation in self.precedence:
            self._preds[position[relation.after_batch_id]].add(
                position[relation.before_batch_id]
            )

        self._deadline = time.monotonic() + self.time_budget
        if self._has_cycle():
            # No permutation can satisfy the mandatory relations; this is a
            # complete (node-less) proof of unsolvability.
            self.cycle_detected = True
            return OrderingOutcome(
                status=STATUS_NO_SOLUTION,
                orderings=[],
                search_complete=True,
                searched_nodes=0,
                notes=self._notes(STATUS_NO_SOLUTION),
            )
        self._dfs(
            start,
            tuple(range(len(self.batches))),
            (),
            frozenset(),
        )

        if len(self.solutions) >= 2:
            # Two different feasible orders are a proof of ambiguity even if a
            # budget ran out afterwards; the search did not need to continue.
            status = STATUS_AMBIGUOUS
        elif self.aborted:
            status = STATUS_SEARCH_LIMIT
        elif len(self.solutions) == 1:
            status = STATUS_UNIQUE
        else:
            status = STATUS_NO_SOLUTION

        search_complete = not self.aborted and status != STATUS_AMBIGUOUS
        return OrderingOutcome(
            status=status,
            orderings=[list(order) for order in self.solutions],
            search_complete=search_complete,
            searched_nodes=self.nodes,
            notes=self._notes(status),
        )

    # -- DFS -------------------------------------------------------------
    def _has_cycle(self) -> bool:
        """Kahn's algorithm over the mandatory precedence relations."""
        unmet = [len(preds) for preds in self._preds]
        successors: list[list[int]] = [[] for _ in self._preds]
        for index, preds in enumerate(self._preds):
            for pred in preds:
                successors[pred].append(index)
        ready = [i for i, count in enumerate(unmet) if count == 0]
        visited = 0
        while ready:
            index = ready.pop()
            visited += 1
            for successor in successors[index]:
                unmet[successor] -= 1
                if unmet[successor] == 0:
                    ready.append(successor)
        return visited != len(self._preds)

    def _dfs(
        self,
        state: SourceState,
        remaining: tuple[int, ...],
        path: tuple[int, ...],
        placed: frozenset[int],
    ) -> None:
        if len(self.solutions) >= 2:
            self.early_stopped = True
            return
        if not remaining:
            assert self._target is not None
            if same_state(state, self._target):
                self.solutions.append([self.batches[i].batch_id for i in path])
            return

        available = [
            index for index in remaining if self._preds[index] <= placed
        ]
        if not available:
            # Every remaining batch waits for a predecessor that also remains:
            # the mandatory relations contain a cycle.
            self.cycle_detected = True
            return

        for index in available:
            if len(self.solutions) >= 2:
                self.early_stopped = True
                return
            if not self._consume_budget():
                return
            batch = self.batches[index]
            try:
                next_state, _ = apply_batch_recording(
                    state, batch, where=f"批次 {batch.batch_id}"
                )
            except InputError:
                # The batch is illegal at this point of the order; no
                # completion of this prefix can make it legal.  Not evidence
                # for or against feasibility, just a pruned branch.
                self.pruned_invalid += 1
                continue
            self._dfs(
                next_state,
                tuple(i for i in remaining if i != index),
                path + (index,),
                placed | {index},
            )

    # -- reporting -------------------------------------------------------
    def _notes(self, status: str) -> list[str]:
        total = len(self.batches)
        if status == STATUS_UNIQUE:
            notes = [
                f"唯一可行顺序：搜索已穷尽，{total} 个批次只有这一种顺序"
                f"（共尝试扩展 {self.nodes} 次）。",
                "该顺序使用全部批次、满足全部先后关系，且终态与权威快照一致。",
            ]
            if self.pruned_invalid:
                notes.append(
                    f"搜索中因批内目标不存在/主键冲突/外键不成立而剪枝 "
                    f"{self.pruned_invalid} 次。"
                )
            return notes
        if status == STATUS_AMBIGUOUS:
            notes = [
                "找到两种不同的可行顺序，判定为歧义（ambiguous）；"
                "找到两种后即停止搜索。",
                "两种顺序都使用全部批次、满足全部先后关系，且终态均与权威快照一致。",
                f"共尝试扩展 {self.nodes} 次；"
                + (
                    "搜索在第二种顺序出现后提前停止，"
                    if self.early_stopped
                    else ""
                )
                + "未穷尽搜索，但两种顺序已足以证明存在歧义。",
            ]
            return notes
        if status == STATUS_SEARCH_LIMIT:
            notes = [
                "搜索预算（节点数或时间）耗尽，仍有未检查的分支；"
                "未查完的分支不能据以宣称唯一或无解。",
                f"共尝试扩展 {self.nodes} 次（节点上限 {self.node_budget}）。",
            ]
            if self.solutions:
                notes.append(
                    f"已找到 {len(self.solutions)} 种可行顺序，但搜索未完成，"
                    "不能据此宣称唯一。"
                )
            else:
                notes.append("尚未找到可行顺序，但搜索未完成，不能据此宣称无解。")
            return notes

        # no_solution
        notes = [
            "已穷尽搜索：不存在使用全部批次、满足全部先后关系且终态等于"
            "权威快照的顺序。",
            f"共尝试扩展 {self.nodes} 次，所有分支均被排除。",
        ]
        if self.cycle_detected:
            notes.append("先后关系存在环，任何顺序都无法同时满足。")
        if self.pruned_invalid:
            notes.append(
                f"搜索中因批内规则（目标不存在/主键冲突/外键不成立）排除前缀 "
                f"{self.pruned_invalid} 次。"
            )
        return notes
