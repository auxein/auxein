import asyncio

import numpy as np
import pytest

from auxein.backend import Array, Backend
from auxein.core import (
    ArrayBatch,
    Batch,
    Cost,
    EvalContext,
    Evaluation,
    EvaluationBatch,
    IdIssuer,
    Objective,
    ProblemSpec,
    StateDict,
    Status,
    StrategyCapabilities,
    StrategyContext,
    validate_state_dict,
)
from auxein.random import RunSeed
from auxein.spaces import Box


def test_capabilities():
    caps = StrategyCapabilities(max_objectives=1, supports_constraints=False, tell_mode="generation")
    assert caps.max_objectives == 1 and caps.tell_mode == "generation"
    assert StrategyCapabilities(None, True, "both").max_objectives is None
    with pytest.raises(ValueError, match="max_objectives"):
        StrategyCapabilities(0, True, "both")
    with pytest.raises(ValueError, match="tell_mode"):
        StrategyCapabilities(1, True, "async")  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        caps.tell_mode = "both"  # type: ignore[misc]


def test_contexts_carry_what_the_driver_provides(backend: Backend):
    seed, issuer = RunSeed(1), IdIssuer()
    strategy_ctx = StrategyContext(seed.stream("strategy", backend=backend), backend, issuer.next)
    assert strategy_ctx.new_id() == 0 and strategy_ctx.new_id() == 1
    assert strategy_ctx.rng.backend == backend

    problem = ProblemSpec(Box(0.0, 1.0, dim=2), (Objective("value"),))
    eval_ctx = EvalContext(
        problem,
        backend,
        lambda cid: seed.stream("evaluation", cid, backend=backend),
        lambda cid: seed.stream("evaluation-batch", cid, backend=backend),
        timeout=5.0,
    )
    assert eval_ctx.problem is problem
    assert eval_ctx.deadline is None and eval_ctx.timeout == 5.0
    a, b, a_again = eval_ctx.rng_for(3), eval_ctx.rng_for(4), eval_ctx.rng_for(3)
    assert backend.to_numpy(a.uniform(3)).tolist() == backend.to_numpy(a_again.uniform(3)).tolist()  # follows the candidate
    assert backend.to_numpy(b.uniform(3)).tolist() != backend.to_numpy(eval_ctx.rng_for(3).uniform(3)).tolist()


class RandomSearchStub:
    """A minimal strategy, to check that the protocols can be implemented and that the pieces compose."""

    capabilities = StrategyCapabilities(max_objectives=1, supports_constraints=False, tell_mode="generation")

    def __init__(self) -> None:
        self.best = float("inf")
        self.step = 0

    def bind(self, problem: ProblemSpec[Array], ctx: StrategyContext) -> None:
        if len(problem.objectives) != 1:
            raise ValueError("this strategy is single-objective")
        self.problem, self.ctx = problem, ctx

    def ask(self, n: int) -> Batch[Array]:
        genomes = self.problem.space.sample_genomes(n, self.ctx.rng, self.ctx.backend)
        batch = ArrayBatch(genomes, [self.ctx.new_id() for _ in range(n)], self.step, "init")
        self.step += 1
        return batch

    def tell(self, results: EvaluationBatch[Array]) -> None:
        values = results.minimisation_matrix(self.problem.objectives, self.ctx.backend)
        self.best = min(self.best, float(self.ctx.backend.to_numpy(values).min()))

    def should_stop(self) -> bool:
        return False

    def state_dict(self) -> StateDict:
        return {"best": self.best, "step": self.step}

    def load_state_dict(self, state: StateDict) -> None:
        self.best, self.step = float(state["best"]), int(state["step"])  # type: ignore[arg-type]


class SphereEvaluator:
    async def evaluate(self, batch: Batch[Array], ctx: EvalContext[Array]) -> EvaluationBatch[Array]:
        array = batch.as_array()
        assert array is not None
        values = ctx.backend.to_numpy((array * array).sum(axis=1))
        return EvaluationBatch(
            [Evaluation(c, Status.OK, {"value": float(v)}, cost=Cost(0.0)) for c, v in zip(batch.candidates, values, strict=True)]
        )


def test_the_core_types_compose_into_an_ask_evaluate_tell_loop(backend: Backend):
    """Not the driver (a later step): just proof that strategy, evaluator and batches fit together on every backend."""
    seed, issuer = RunSeed(5), IdIssuer()
    problem = ProblemSpec(Box(-5.0, 5.0, dim=4), (Objective("value"),))
    strategy = RandomSearchStub()
    strategy.bind(problem, StrategyContext(seed.stream("strategy", backend=backend), backend, issuer.next))
    ctx = EvalContext(
        problem,
        backend,
        lambda cid: seed.stream("evaluation", cid, backend=backend),
        lambda cid: seed.stream("evaluation-batch", cid, backend=backend),
    )

    for _ in range(3):
        batch = strategy.ask(8)
        results = asyncio.run(SphereEvaluator().evaluate(batch, ctx))
        assert [e.candidate.id for e in results] == [c.id for c in batch.candidates]
        strategy.tell(results)

    assert issuer.issued == 24 and strategy.step == 3
    assert np.isfinite(strategy.best) and strategy.best >= 0
    state = strategy.state_dict()
    validate_state_dict(state)
    restored = RandomSearchStub()
    restored.load_state_dict(state)
    assert restored.best == strategy.best and restored.step == 3


def test_a_single_objective_strategy_rejects_two_objectives_at_bind():
    problem = ProblemSpec(Box(0.0, 1.0, dim=2), (Objective("a"), Objective("b")))
    strategy = RandomSearchStub()
    ctx = StrategyContext(RunSeed(1).stream("strategy"), Backend(), IdIssuer().next)
    with pytest.raises(ValueError, match="single-objective"):
        strategy.bind(problem, ctx)
