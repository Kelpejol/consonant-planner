"""
Terra Planner - Data Models

This module defines all Pydantic models used throughout the planning system.
These models provide runtime validation, type safety, and clear documentation
of data structures flowing through the planning cognitive loop.

Models are divided into:
1. Planning State - The core state that flows through the LangGraph
2. Planning Artifacts - Intermediate reasoning products
3. Agent Models - Agent capability and behavioral representations
4. Result Models - Structured results from analysis phases
"""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS - Type-safe categorical values
# ============================================================================

class ConfidenceLevel(str, Enum):
    """
    Represents the planner's confidence in a conclusion or assumption.
    
    This is qualitative, not numeric - it reflects the planner's epistemic
    state based on evidence quality and consistency.
    """
    VERY_LOW = "very_low"      # Highly uncertain, speculative
    LOW = "low"                # Some evidence but weak
    MEDIUM = "medium"          # Moderate supporting evidence
    HIGH = "high"              # Strong, consistent evidence
    VERY_HIGH = "very_high"    # Near-certain, multiple confirming sources


class StrategyHealth(str, Enum):
    """
    Assessment of the current strategy's viability.
    
    This determines whether the planner should continue, branch, or replan.
    """
    VIABLE = "viable"              # Strategy is working, continue
    WEAKENING = "weakening"        # Results don't reduce uncertainty much
    QUESTIONABLE = "questionable"  # Assumptions starting to break
    FAILED = "failed"              # Strategy clearly won't reach goal


class PlanningPhase(str, Enum):
    """The current cognitive phase of the planner."""
    UNDERSTAND = "understand"      # Goal interpretation
    DECOMPOSE = "decompose"        # Sub-goal identification
    MATCH = "match"                # Agent capability matching
    PLAN = "plan"                  # Execution structuring
    AWAIT = "await"                # Waiting for results
    ANALYZE = "analyze"            # Result interpretation
    DECIDE = "decide"              # Strategy decision
    COMPLETE = "complete"          # Goal satisfied or impossible


class ExecutionMode(str, Enum):
    """How a plan step should be executed."""
    SEQUENTIAL = "sequential"      # Must wait for dependencies
    PARALLEL = "parallel"          # Can run independently
    CONDITIONAL = "conditional"    # Only if condition met


# ============================================================================
# AGENT MODELS - Agent capability and behavior representation
# ============================================================================

class AgentCapability(BaseModel):
    """
    Represents what an agent can contribute to solving sub-goals.
    
    This is semantic, not technical - it describes cognitive/operational
    contributions, not implementation details like "runs Python code".
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    agent_id: str = Field(
        ..., 
        description="Unique identifier for the agent"
    )
    
    capability_type: str = Field(
        ...,
        description="Semantic capability (e.g., 'forensic_analysis', 'code_generation')"
    )
    
    description: str = Field(
        ...,
        description="What this agent can contribute in natural language"
    )
    
    strengths: List[str] = Field(
        default_factory=list,
        description="Known areas where this agent excels"
    )
    
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Known limitations or failure modes"
    )
    
    typical_unknowns_addressed: List[str] = Field(
        default_factory=list,
        description="Types of unknowns this agent typically helps resolve"
    )


class AgentBehaviorHistory(BaseModel):
    """
    Tracks observed behavior of an agent over time.
    
    This is the planner's learned model of the agent - not just statistics,
    but semantic understanding of how the agent behaves under different conditions.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    agent_id: str
    
    total_invocations: int = Field(
        default=0,
        description="Number of times this agent has been used in planning"
    )
    
    successful_resolutions: int = Field(
        default=0,
        description="Times the agent successfully resolved an unknown"
    )
    
    partial_results_count: int = Field(
        default=0,
        description="Times the agent gave incomplete but useful info"
    )
    
    noise_introductions: int = Field(
        default=0,
        description="Times the agent added unhelpful complexity"
    )
    
    overconfident_conclusions: int = Field(
        default=0,
        description="Times the agent was confident but wrong"
    )
    
    behavioral_patterns: List[str] = Field(
        default_factory=list,
        description="Semantic observations (e.g., 'good for quick triage')"
    )
    
    trust_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Overall trust in this agent's outputs"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this history was last updated"
    )


# ============================================================================
# PLANNING ARTIFACTS - Intermediate reasoning products
# ============================================================================

class Unknown(BaseModel):
    """
    Represents a piece of missing information that blocks goal achievement.
    
    The planner is driven by identifying and resolving unknowns.
    This is the fundamental unit of planning - not steps, but information gaps.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    unknown_id: str = Field(
        ...,
        description="Unique identifier for this unknown"
    )
    
    description: str = Field(
        ...,
        description="What we don't know (natural language)"
    )
    
    importance: ConfidenceLevel = Field(
        ...,
        description="How critical this unknown is to goal achievement"
    )
    
    related_sub_goal: Optional[str] = Field(
        None,
        description="Which sub-goal this unknown blocks"
    )
    
    potential_resolvers: List[str] = Field(
        default_factory=list,
        description="Agent IDs that might address this unknown"
    )


class SubGoal(BaseModel):
    """
    A logical objective that must be satisfied for the main goal to be achieved.
    
    Sub-goals are NOT procedural steps - they are logical requirements.
    Good: "Determine failure category"
    Bad: "Check logs"
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    sub_goal_id: str = Field(
        ...,
        description="Unique identifier"
    )
    
    description: str = Field(
        ...,
        description="What needs to be true/known (logical, not procedural)"
    )
    
    unknowns: List[Unknown] = Field(
        default_factory=list,
        description="Information gaps that block this sub-goal"
    )
    
    assumptions: List[str] = Field(
        default_factory=list,
        description="What we're assuming to be true"
    )
    
    satisfaction_criteria: str = Field(
        ...,
        description="How we know this sub-goal is satisfied"
    )
    
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.VERY_LOW,
        description="Current confidence this sub-goal is achievable"
    )


class PlanStep(BaseModel):
    """
    A concrete action to be taken, derived from reasoning about unknowns.
    
    This is what gets returned in the final plan - but it's generated from
    deep reasoning about unknowns, capabilities, and dependencies.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(
        ...,
        description="Unique step identifier"
    )
    
    description: str = Field(
        ...,
        description="What should be done (clear, actionable)"
    )
    
    agent_selector: str = Field(
        ...,
        description="Type of agent required (capability type, not agent ID)"
    )
    
    sub_goal_addressed: str = Field(
        ...,
        description="Which sub-goal this step helps satisfy"
    )
    
    unknowns_targeted: List[str] = Field(
        default_factory=list,
        description="Unknown IDs this step attempts to resolve"
    )
    
    dependencies: List[str] = Field(
        default_factory=list,
        description="Step IDs that must complete before this"
    )
    
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SEQUENTIAL,
        description="How this step should be executed"
    )
    
    rationale: str = Field(
        ...,
        description="Why this step was chosen (for learning/debugging)"
    )
    
    expected_outcome: str = Field(
        ...,
        description="What information we expect to gain"
    )


class Strategy(BaseModel):
    """
    The current hypothesis about how to achieve the goal.
    
    A strategy is a coherent set of assumptions that guide all planning decisions.
    When assumptions break, strategies are abandoned and replanned.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    strategy_id: str
    
    hypothesis: str = Field(
        ...,
        description="Core assumption about path to goal (e.g., 'issue is config-related')"
    )
    
    assumptions: List[str] = Field(
        default_factory=list,
        description="What must be true for this strategy to work"
    )
    
    sub_goals: List[SubGoal] = Field(
        default_factory=list,
        description="Sub-goals derived from this strategy"
    )
    
    health: StrategyHealth = Field(
        default=StrategyHealth.VIABLE,
        description="Current assessment of strategy viability"
    )
    
    iterations: int = Field(
        default=0,
        description="How many plan-execute-analyze cycles"
    )
    
    resolved_unknowns: List[str] = Field(
        default_factory=list,
        description="Unknown IDs resolved under this strategy"
    )
    
    confidence_trajectory: List[ConfidenceLevel] = Field(
        default_factory=list,
        description="How confidence has evolved over iterations"
    )


# ============================================================================
# RESULT MODELS - Structured analysis of agent results
# ============================================================================

class ResultAnalysis(BaseModel):
    """
    The planner's interpretation of results from agents.
    
    This is not just data - it's semantic understanding of what the
    results mean for the current strategy and unknowns.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    step_id: str = Field(
        ...,
        description="Which step produced these results"
    )
    
    agent_id: str = Field(
        ...,
        description="Which agent executed the step"
    )
    
    raw_results: str = Field(
        ...,
        description="The actual output from the agent"
    )
    
    unknowns_resolved: List[str] = Field(
        default_factory=list,
        description="Unknown IDs that are now answered"
    )
    
    unknowns_introduced: List[Unknown] = Field(
        default_factory=list,
        description="New unknowns discovered from results"
    )
    
    assumptions_validated: List[str] = Field(
        default_factory=list,
        description="Which assumptions were confirmed"
    )
    
    assumptions_invalidated: List[str] = Field(
        default_factory=list,
        description="Which assumptions were refuted"
    )
    
    confidence_in_results: ConfidenceLevel = Field(
        ...,
        description="How much to trust these results"
    )
    
    interpretation: str = Field(
        ...,
        description="What the planner concluded from these results"
    )
    
    suggests_replanning: bool = Field(
        default=False,
        description="Whether results indicate strategy should be reconsidered"
    )


# ============================================================================
# PLANNING STATE - Core state flowing through LangGraph
# ============================================================================

class PlannerContext(BaseModel):
    """
    The accumulated context that persists across planning iterations.
    
    This is the planner's memory - what it knows, what it doesn't know,
    and what it's learned about agents and strategies.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Core knowledge
    known_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Things we know to be true (key: fact_id, value: fact)"
    )
    
    resolved_unknowns: Dict[str, str] = Field(
        default_factory=dict,
        description="Unknown ID → resolution mapping"
    )
    
    remaining_unknowns: List[Unknown] = Field(
        default_factory=list,
        description="Unknowns still blocking progress"
    )
    
    # Strategy tracking
    active_strategy: Optional[Strategy] = Field(
        None,
        description="Current strategic hypothesis"
    )
    
    abandoned_strategies: List[Strategy] = Field(
        default_factory=list,
        description="Previous strategies that didn't work"
    )
    
    # Agent learning
    agent_capabilities: Dict[str, AgentCapability] = Field(
        default_factory=dict,
        description="Known agent capabilities (agent_id → capability)"
    )
    
    agent_history: Dict[str, AgentBehaviorHistory] = Field(
        default_factory=dict,
        description="Learned behavioral patterns (agent_id → history)"
    )
    
    # Execution tracking
    completed_steps: List[PlanStep] = Field(
        default_factory=list,
        description="Steps that have been executed"
    )
    
    result_history: List[ResultAnalysis] = Field(
        default_factory=list,
        description="Analysis of all results received"
    )


class PlannerState(BaseModel):
    """
    The complete state that flows through the LangGraph cognitive loop.
    
    This is the definitive state schema - TypedDict is used in LangGraph
    for performance, but this Pydantic model defines the contract.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input
    goal: str = Field(
        ...,
        description="The original user goal (may be refined)"
    )
    
    workflow_id: str = Field(
        ...,
        description="Unique identifier for this planning session"
    )
    
    # Current phase
    current_phase: PlanningPhase = Field(
        default=PlanningPhase.UNDERSTAND,
        description="Which cognitive phase we're in"
    )
    
    # Planning artifacts
    context: PlannerContext = Field(
        default_factory=PlannerContext,
        description="The planner's accumulated knowledge"
    )
    
    current_plan: List[PlanStep] = Field(
        default_factory=list,
        description="The current plan being executed"
    )
    
    # Iteration management
    iteration_count: int = Field(
        default=0,
        description="How many complete cycles through the loop"
    )
    
    max_iterations: int = Field(
        default=10,
        description="Max iterations before forced termination"
    )
    
    # LLM interactions
    llm_messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Message history for LLM context"
    )
    
    # Decision tracking
    last_decision: Optional[str] = Field(
        None,
        description="The most recent strategic decision"
    )
    
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Explanations of key decisions (for debugging/learning)"
    )
    
    # Completion
    goal_satisfied: bool = Field(
        default=False,
        description="Whether the goal has been achieved"
    )
    
    goal_impossible: bool = Field(
        default=False,
        description="Whether we've determined the goal cannot be reached"
    )
    
    final_reasoning: Optional[str] = Field(
        None,
        description="Explanation of why we stopped planning"
    )
    
    @field_validator('iteration_count')
    @classmethod
    def validate_iteration_count(cls, v: int) -> int:
        """Ensure iteration count is non-negative."""
        if v < 0:
            raise ValueError("iteration_count must be non-negative")
        return v
    
    @field_validator('max_iterations')
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Ensure max_iterations is positive."""
        if v <= 0:
            raise ValueError("max_iterations must be positive")
        return v


# ============================================================================
# UTILITY MODELS
# ============================================================================

class PlanningMetrics(BaseModel):
    """
    Metrics for monitoring planner performance.
    
    These are not used in the cognitive loop but are useful for
    observability and continuous improvement.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    workflow_id: str
    total_iterations: int
    total_steps_generated: int
    total_unknowns_resolved: int
    strategies_attempted: int
    final_confidence: ConfidenceLevel
    time_to_completion_seconds: float
    llm_calls_made: int
    goal_achieved: bool