"""
Budget-to-config generator for DiffSparseKV.

This module maps a target average sparsity budget to a valid differential
sparsity configuration. The current implementation uses template families where
all but one sparsity level are fixed and the remaining level is solved
analytically to match the requested budget.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BudgetTemplate:
    name: str
    target_distribution: List[float]
    sparsity_levels: List[Optional[float]]
    description: str

    def resolve(self, target_budget: float) -> List[float]:
        if not 0.0 <= target_budget <= 1.0:
            raise ValueError(f"target_budget must be in [0, 1], got {target_budget}")

        free_indices = [idx for idx, value in enumerate(self.sparsity_levels) if value is None]
        if len(free_indices) != 1:
            raise ValueError(
                f"Template '{self.name}' must contain exactly one free sparsity level, found {len(free_indices)}"
            )

        free_idx = free_indices[0]
        free_weight = self.target_distribution[free_idx]
        fixed_budget = sum(
            weight * level
            for weight, level in zip(self.target_distribution, self.sparsity_levels)
            if level is not None
        )

        min_budget = fixed_budget
        max_budget = fixed_budget + free_weight
        if not (min_budget - 1e-9 <= target_budget <= max_budget + 1e-9):
            raise ValueError(
                f"Budget {target_budget:.3f} is outside the feasible range "
                f"[{min_budget:.3f}, {max_budget:.3f}] for template '{self.name}'"
            )

        if free_weight <= 0.0:
            raise ValueError(f"Template '{self.name}' has non-positive free weight")

        free_level = (target_budget - fixed_budget) / free_weight
        free_level = max(0.0, min(1.0, free_level))

        resolved = [free_level if value is None else float(value) for value in self.sparsity_levels]
        if any(resolved[idx] > resolved[idx + 1] for idx in range(len(resolved) - 1)):
            raise ValueError(
                f"Resolved sparsity levels are not non-decreasing for template '{self.name}': {resolved}"
            )
        return resolved


@dataclass(frozen=True)
class ResolvedBudgetConfig:
    template_name: str
    target_budget: float
    target_distribution: List[float]
    sparsity_levels: List[float]
    expected_sparsity: float
    description: str


BUILTIN_BUDGET_TEMPLATES: Dict[str, BudgetTemplate] = {
    "default_3level": BudgetTemplate(
        name="default_3level",
        target_distribution=[0.05, 0.75, 0.20],
        sparsity_levels=[0.0, None, 1.0],
        description="5% dense, 75% adaptive middle bucket, 20% eviction bucket.",
    ),
    "conservative_3level": BudgetTemplate(
        name="conservative_3level",
        target_distribution=[0.10, 0.70, 0.20],
        sparsity_levels=[0.0, None, 1.0],
        description="10% dense, 70% adaptive middle bucket, 20% eviction bucket.",
    ),
    "low_evict_3level": BudgetTemplate(
        name="low_evict_3level",
        target_distribution=[0.10, 0.80, 0.10],
        sparsity_levels=[0.0, None, 1.0],
        description="10% dense, 80% adaptive middle bucket, 10% eviction bucket.",
    ),
}


def list_budget_templates() -> Dict[str, str]:
    return {name: template.description for name, template in BUILTIN_BUDGET_TEMPLATES.items()}


def resolve_budget_config(
    target_budget: float,
    template_name: str = "default_3level",
) -> ResolvedBudgetConfig:
    if template_name not in BUILTIN_BUDGET_TEMPLATES:
        available = ", ".join(sorted(BUILTIN_BUDGET_TEMPLATES))
        raise ValueError(f"Unknown budget template '{template_name}'. Available: {available}")

    template = BUILTIN_BUDGET_TEMPLATES[template_name]
    resolved_levels = template.resolve(target_budget)
    expected = sum(
        weight * level for weight, level in zip(template.target_distribution, resolved_levels)
    )
    return ResolvedBudgetConfig(
        template_name=template.name,
        target_budget=target_budget,
        target_distribution=list(template.target_distribution),
        sparsity_levels=resolved_levels,
        expected_sparsity=expected,
        description=template.description,
    )
