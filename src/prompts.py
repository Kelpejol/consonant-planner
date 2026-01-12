"""
Terra Planner - LLM Prompts

This module contains all prompts used by the planner at each cognitive phase.

Each prompt is carefully designed to:
1. Focus on reasoning, not execution
2. Work with unknowns and assumptions, not just steps
3. Maintain epistemic humility (confidence levels)
4. Enable learning from agent behavior
5. Support strategic thinking (replanning, branching)

Prompts follow the principle: Good planners ask "what do I not know?"
rather than "what steps should I run?"
"""

from typing import List, Dict, Any
from src.models import (
    SubGoal, Unknown, AgentCapability, PlanStep, 
    ResultAnalysis, Strategy, PlannerContext
)


# ============================================================================
# PHASE 1: UNDERSTAND (Goal Interpretation)
# ============================================================================

def create_understand_prompt(
    goal: str,
    context: Dict[str, Any]
) -> str:
    """
    Generate prompt for the UNDERSTAND phase.
    
    This phase reframes the goal into a structured understanding of:
    - Real objectives (what does success mean?)
    - Constraints (what must be avoided?)
    - Unknowns (what's missing to even start?)
    - Compound vs single goals
    
    NOT: Generate steps
    BUT: Deeply understand what's being asked
    
    Args:
        goal: The user's original goal statement
        context: Any additional context provided
        
    Returns:
        Formatted prompt for the LLM
    """
    context_str = "\n".join([f"  - {k}: {v}" for k, v in context.items()])
    
    return f"""You are a strategic planning expert analyzing a goal to determine its real requirements.

GOAL TO ANALYZE:
"{goal}"

AVAILABLE CONTEXT:
{context_str if context else "  (No additional context provided)"}

YOUR TASK:
Do NOT generate steps or actions. Instead, perform deep goal analysis:

1. OBJECTIVE CLARIFICATION
   - What is the real objective? (Sometimes the stated goal is a symptom, not the root need)
   - Is this a single goal or compound? If compound, break into separate objectives.
   - What does "success" mean for this goal? Be specific.

2. CONSTRAINT IDENTIFICATION
   - What must be avoided? (e.g., "don't break unrelated systems")
   - What resources/time limitations exist?
   - Are there unstated assumptions in the goal statement?

3. INITIAL UNKNOWNS
   - What critical information is missing right now?
   - What would we need to know to even begin planning?
   - What ambiguities need resolution?

4. GOAL REFINEMENT
   - Restate the goal in a more precise form
   - If the goal is misleading, propose a reframing

OUTPUT REQUIREMENTS:
Provide a structured JSON with:
{{
  "refined_goal": "Precise restatement of the goal",
  "objectives": ["List of distinct objectives"],
  "success_criteria": "What success looks like",
  "constraints": ["Things to avoid or respect"],
  "initial_unknowns": ["Critical missing information"],
  "assumptions_in_goal": ["Implicit assumptions to validate"],
  "confidence": "high|medium|low - how well-defined is this goal?"
}}

REMEMBER: You are not planning yet - you are understanding what planning must achieve.
"""


# ============================================================================
# PHASE 2: DECOMPOSE (Sub-goal Identification)
# ============================================================================

def create_decompose_prompt(
    refined_goal: str,
    objectives: List[str],
    unknowns: List[Unknown],
    context: PlannerContext
) -> str:
    """
    Generate prompt for the DECOMPOSE phase.
    
    This phase identifies logical sub-goals (not procedural steps).
    
    Key principle: Sub-goals are conditions that must be true for the
    goal to be satisfied - they are logical, not procedural.
    
    Good: "Failure category is determined"
    Bad: "Check logs"
    
    Args:
        refined_goal: The goal as understood in UNDERSTAND phase
        objectives: List of distinct objectives
        unknowns: Current unknowns blocking progress
        context: The planner's accumulated context
        
    Returns:
        Formatted prompt for the LLM
    """
    unknowns_str = "\n".join([
        f"  - {u.description} (importance: {u.importance})"
        for u in unknowns
    ])
    
    objectives_str = "\n".join([f"  - {obj}" for obj in objectives])
    
    known_facts_str = "\n".join([
        f"  - {k}: {v}" for k, v in context.known_facts.items()
    ]) if context.known_facts else "  (No facts established yet)"
    
    return f"""You are decomposing a goal into logical sub-goals (NOT steps).

REFINED GOAL:
{refined_goal}

DISTINCT OBJECTIVES:
{objectives_str}

CURRENT UNKNOWNS:
{unknowns_str}

KNOWN FACTS:
{known_facts_str}

YOUR TASK:
Identify SUB-GOALS that must be satisfied for the main goal to be achieved.

CRITICAL DISTINCTION:
❌ BAD (procedural): "Check logs", "Run diagnostics", "Query database"
✅ GOOD (logical): "Failure cause is identified", "Scope of impact is known", "Root issue is isolated"

REASONING PROCESS:
1. For each objective, ask: "What must be TRUE for this objective to be satisfied?"
2. Don't think about HOW to make it true - think about WHAT must be true
3. Identify dependencies: which sub-goals depend on others?
4. For each sub-goal, identify the unknowns that block it

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "sub_goals": [
    {{
      "id": "sg-1",
      "description": "Logical condition that must be satisfied",
      "unknowns_blocking": ["List of unknowns that prevent this"],
      "dependencies": ["Other sub-goal IDs this depends on"],
      "assumptions": ["What we're assuming about this sub-goal"],
      "satisfaction_criteria": "How we know this sub-goal is met"
    }}
  ],
  "parallel_groups": [
    ["sg-1", "sg-2"]  // Sub-goals that can be pursued independently
  ],
  "sequential_constraints": [
    {{"dependent": "sg-3", "requires": ["sg-1"]}}
  ],
  "reasoning": "Explanation of the decomposition logic"
}}

REMEMBER: We want to know WHAT must be true, not HOW to make it true.
"""


# ============================================================================
# PHASE 3: MATCH (Agent Capability Matching)
# ============================================================================

def create_match_prompt(
    sub_goals: List[SubGoal],
    unknowns: List[Unknown],
    agent_capabilities: Dict[str, AgentCapability],
    agent_history: Dict[str, Any],
    context: PlannerContext
) -> str:
    """
    Generate prompt for the MATCH phase.
    
    This phase matches unknowns to agent capabilities, considering:
    - What agents can contribute (semantic, not technical)
    - Past behavior (trust, reliability, patterns)
    - Whether to use multiple agents for validation
    
    Args:
        sub_goals: The logical sub-goals to address
        unknowns: Specific unknowns to resolve
        agent_capabilities: Known agent capabilities
        agent_history: Learned behavioral patterns
        context: Planner context
        
    Returns:
        Formatted prompt for the LLM
    """
    sub_goals_str = "\n".join([
        f"  - [{sg.sub_goal_id}] {sg.description}\n"
        f"    Unknowns: {[u.description for u in sg.unknowns]}\n"
        f"    Assumptions: {sg.assumptions}"
        for sg in sub_goals
    ])
    
    agents_str = "\n".join([
        f"  - [{cap.agent_id}] {cap.capability_type}:\n"
        f"    Description: {cap.description}\n"
        f"    Strengths: {cap.strengths}\n"
        f"    Weaknesses: {cap.weaknesses}\n"
        f"    Typical unknowns: {cap.typical_unknowns_addressed}\n"
        f"    Trust level: {agent_history.get(cap.agent_id, {}).get('trust_level', 'medium')}\n"
        f"    Behavioral notes: {agent_history.get(cap.agent_id, {}).get('behavioral_patterns', [])}"
        for cap in agent_capabilities.values()
    ])
    
    unknowns_str = "\n".join([
        f"  - [{u.unknown_id}] {u.description} (importance: {u.importance})"
        for u in unknowns
    ])
    
    return f"""You are matching unknowns to agent capabilities based on semantic fit and historical behavior.

SUB-GOALS TO ADDRESS:
{sub_goals_str}

CURRENT UNKNOWNS:
{unknowns_str}

AVAILABLE AGENTS:
{agents_str if agents_str else "  (No agents registered yet - this is likely a test scenario)"}

YOUR TASK:
For each unknown, determine which agent(s) can best contribute to resolving it.

MATCHING CRITERIA:
1. Semantic Fit: Does the agent's capability type match the unknown's nature?
2. Historical Performance: Has this agent performed well on similar unknowns?
3. Trust Level: How confident are we in this agent's outputs?
4. Complementarity: Should multiple agents be used for validation?

IMPORTANT CONSIDERATIONS:
- Some unknowns may benefit from multiple agents (parallel for speed, sequential for validation)
- If an agent "often gives partial answers", consider pairing with a verifier
- If an agent is "creative but speculative", use for hypothesis generation, not confirmation
- Unknown importance should influence agent selection (high importance → high trust agents)

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "matches": [
    {{
      "unknown_id": "unk-1",
      "sub_goal_id": "sg-1",
      "primary_agent": "agent_id or capability_type",
      "supporting_agents": ["Optional: other agents for validation"],
      "rationale": "Why this agent/these agents?",
      "execution_strategy": "parallel|sequential|conditional",
      "confidence": "high|medium|low"
    }}
  ],
  "parallel_opportunities": [
    {{"unknowns": ["unk-1", "unk-2"], "reason": "Independent, no conflicts"}}
  ],
  "validation_recommendations": [
    {{"unknown": "unk-3", "strategy": "Use agent X for initial, agent Y for verification"}}
  ],
  "reasoning": "Overall matching strategy explanation"
}}

REMEMBER: You're reasoning about WHO can contribute WHAT, informed by past behavior.
"""


# ============================================================================
# PHASE 4: PLAN (Execution Structuring)
# ============================================================================

def create_plan_prompt(
    sub_goals: List[SubGoal],
    matches: List[Dict[str, Any]],
    parallel_groups: List[List[str]],
    context: PlannerContext
) -> str:
    """
    Generate prompt for the PLAN phase.
    
    This phase structures the execution: which steps run in what order,
    which can parallelize, what dependencies exist.
    
    This is where we finally generate concrete steps - but they're deeply
    informed by all the prior reasoning about unknowns and capabilities.
    
    Args:
        sub_goals: The sub-goals to address
        matches: Unknown-to-agent matches from MATCH phase
        parallel_groups: Groups of sub-goals that can run concurrently
        context: Planner context
        
    Returns:
        Formatted prompt for the LLM
    """
    sub_goals_str = "\n".join([
        f"  - [{sg.sub_goal_id}] {sg.description}"
        for sg in sub_goals
    ])
    
    matches_str = "\n".join([
        f"  - Unknown '{m.get('unknown_id')}' → Agent '{m.get('primary_agent')}'\n"
        f"    Rationale: {m.get('rationale')}\n"
        f"    Strategy: {m.get('execution_strategy')}"
        for m in matches
    ])
    
    parallel_str = "\n".join([
        f"  - Can run together: {group}"
        for group in parallel_groups
    ])
    
    return f"""You are structuring concrete execution steps based on sub-goals and agent matches.

SUB-GOALS:
{sub_goals_str}

AGENT MATCHES:
{matches_str}

PARALLELIZATION OPPORTUNITIES:
{parallel_str if parallel_groups else "  (None identified)"}

YOUR TASK:
Generate concrete, executable steps that will resolve unknowns and satisfy sub-goals.

STEP DESIGN PRINCIPLES:
1. Each step should target specific unknowns
2. Steps should be clear about expected outcomes
3. Dependencies should be explicit (can't run until X completes)
4. Parallel steps should truly be independent
5. Include rationale for why each step exists

EXECUTION MODES:
- PARALLEL: Steps that can run simultaneously (no shared state, independent unknowns)
- SEQUENTIAL: Steps with dependencies (B needs results from A)
- CONDITIONAL: Steps that only run if a condition is met

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "steps": [
    {{
      "id": "step-1",
      "description": "Clear, actionable description",
      "agent_selector": "capability_type (e.g., 'log_analyzer')",
      "sub_goal_addressed": "sg-1",
      "unknowns_targeted": ["unk-1", "unk-2"],
      "dependencies": ["step-0"],  // Empty if no dependencies
      "execution_mode": "parallel|sequential|conditional",
      "condition": "If conditional, what condition?",
      "rationale": "Why this step? What will it tell us?",
      "expected_outcome": "What information we expect to gain"
    }}
  ],
  "execution_graph": {{
    "parallel_batches": [
      ["step-1", "step-2"],  // Can run together
      ["step-3"]  // Runs after batch 1
    ]
  }},
  "reasoning": "Overall plan structure explanation",
  "risks": ["Potential issues with this plan"],
  "early_abort_conditions": ["Signals that would invalidate this plan"]
}}

REMEMBER: Steps are informed by unknowns and capabilities, not guessed blind.
"""


# ============================================================================
# PHASE 5: ANALYZE (Result Interpretation)
# ============================================================================

def create_analyze_prompt(
    step_results: Dict[str, str],
    original_plan: List[PlanStep],
    current_strategy: Strategy,
    context: PlannerContext
) -> str:
    """
    Generate prompt for the ANALYZE phase.
    
    This is where intelligence shows: interpreting results to:
    - Determine which unknowns are now resolved
    - Identify new unknowns introduced
    - Validate or invalidate assumptions
    - Assess result quality and trustworthiness
    
    NOT: Just report what agents said
    BUT: Reason about what it means for our strategy
    
    Args:
        step_results: Results from executed steps (step_id → result)
        original_plan: The steps that were executed
        current_strategy: The active strategic hypothesis
        context: Planner context
        
    Returns:
        Formatted prompt for the LLM
    """
    results_str = "\n\n".join([
        f"STEP {step_id}:\n"
        f"Description: {next((s.description for s in original_plan if s.id == step_id), 'Unknown')}\n"
        f"Target unknowns: {next((s.unknowns_targeted for s in original_plan if s.id == step_id), [])}\n"
        f"Expected outcome: {next((s.expected_outcome for s in original_plan if s.id == step_id), 'Unknown')}\n"
        f"RESULT:\n{result}\n"
        f"{'='*60}"
        for step_id, result in step_results.items()
    ])
    
    strategy_str = f"""
    Hypothesis: {current_strategy.hypothesis}
    Assumptions: {current_strategy.assumptions}
    Health: {current_strategy.health}
    Iterations: {current_strategy.iterations}
    """
    
    unknowns_str = "\n".join([
        f"  - [{u.unknown_id}] {u.description}"
        for u in context.remaining_unknowns
    ])
    
    return f"""You are analyzing results from agents to determine what we've learned and what it means.

CURRENT STRATEGY:
{strategy_str}

UNKNOWNS WE WERE TRYING TO RESOLVE:
{unknowns_str}

RESULTS RECEIVED:
{results_str}

YOUR TASK:
Deeply analyze these results - don't just report them, INTERPRET them.

ANALYSIS FRAMEWORK:
1. UNKNOWN RESOLUTION
   - Which unknowns are now answered? (Be specific about what we learned)
   - Which unknowns remain open?
   - Did results introduce NEW unknowns?

2. ASSUMPTION VALIDATION
   - Which strategy assumptions were confirmed?
   - Which were refuted or cast into doubt?
   - Are we more or less confident in our hypothesis?

3. RESULT QUALITY
   - How confident should we be in these results?
   - Did agents provide partial/complete/noisy information?
   - Do results from different agents agree or contradict?

4. STRATEGIC IMPLICATIONS
   - Does this change our path forward?
   - Are results informative or just noise?
   - Should we continue, branch, or replan?

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "unknowns_resolved": [
    {{"id": "unk-1", "resolution": "What we learned", "confidence": "high|medium|low"}}
  ],
  "unknowns_still_open": ["unk-2"],
  "new_unknowns": [
    {{"description": "New question raised", "importance": "high|medium|low"}}
  ],
  "assumptions_validated": ["assumption-1"],
  "assumptions_invalidated": ["assumption-2"],
  "result_coherence": "Do results tell a consistent story?",
  "confidence_change": "Did confidence increase, decrease, or stay same?",
  "information_gain": "high|medium|low - how much did we learn?",
  "strategic_assessment": {{
    "continue_viable": true/false,
    "branch_recommended": true/false,
    "replan_required": true/false,
    "reasoning": "Why?"
  }},
  "interpretation": "Overall synthesis of what results mean"
}}

CRITICAL: Don't trust results blindly. Consider agent history and result coherence.
"""


# ============================================================================
# PHASE 6: DECIDE (Strategic Decision)
# ============================================================================

def create_decide_prompt(
    analysis: ResultAnalysis,
    current_strategy: Strategy,
    remaining_unknowns: List[Unknown],
    iteration_count: int,
    context: PlannerContext
) -> str:
    """
    Generate prompt for the DECIDE phase.
    
    This phase determines: continue current strategy, branch into multiple
    hypotheses, or abandon and replan.
    
    This is pure strategic reasoning, not reactive to failure.
    
    Args:
        analysis: The result analysis from ANALYZE phase
        current_strategy: Current strategic hypothesis
        remaining_unknowns: What we still don't know
        iteration_count: How many iterations we've done
        context: Planner context
        
    Returns:
        Formatted prompt for the LLM
    """
    strategy_str = f"""
    Hypothesis: {current_strategy.hypothesis}
    Health: {current_strategy.health}
    Iterations: {current_strategy.iterations}
    Resolved unknowns: {len(current_strategy.resolved_unknowns)}
    Confidence trajectory: {current_strategy.confidence_trajectory}
    """
    
    unknowns_str = "\n".join([
        f"  - {u.description} (importance: {u.importance})"
        for u in remaining_unknowns
    ])
    
    abandoned_str = "\n".join([
        f"  - {s.hypothesis} (abandoned after {s.iterations} iterations)"
        for s in context.abandoned_strategies
    ]) if context.abandoned_strategies else "  (None yet)"
    
    return f"""You are making a strategic decision about how to proceed.

CURRENT STRATEGY:
{strategy_str}

RECENT ANALYSIS:
Unknowns resolved: {len(analysis.unknowns_resolved)}
Unknowns introduced: {len(analysis.unknowns_introduced)}
Assumptions invalidated: {analysis.assumptions_invalidated}
Confidence in results: {analysis.confidence_in_results}
Strategic assessment: {analysis.suggests_replanning}

REMAINING UNKNOWNS:
{unknowns_str if unknowns_str else "  (All unknowns resolved!)"}

ITERATION CONTEXT:
Current iteration: {iteration_count}
Previously abandoned strategies:
{abandoned_str}

YOUR TASK:
Decide whether to CONTINUE, BRANCH, or REPLAN - and explain why.

DECISION CRITERIA:

CONTINUE when:
- Strategy is working (unknowns decreasing)
- Assumptions still hold
- Confidence increasing
- Clear path forward

BRANCH when:
- Multiple plausible explanations exist
- Results suggest parallel hypotheses
- Cost of exploration is acceptable
- Early disambiguation is hard

REPLAN when:
- Core assumptions collapsed
- Strategy not reducing uncertainty
- Repeating ourselves (redundancy)
- Results consistently contradict hypothesis

GOAL COMPLETION:
- If no unknowns remain AND confidence is high → Goal satisfied
- If strategy has failed repeatedly AND no alternatives → Goal impossible

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "decision": "continue|branch|replan|complete",
  "reasoning": "Detailed explanation of why",
  "strategy_health_assessment": "viable|weakening|questionable|failed",
  "confidence_in_direction": "high|medium|low",
  "next_actions": {{
    "if_continue": "What to do next under current strategy",
    "if_branch": "What hypotheses to pursue in parallel",
    "if_replan": "How to reframe the problem",
    "if_complete": "Why we're done"
  }},
  "goal_status": {{
    "satisfied": true/false,
    "impossible": true/false,
    "explanation": "Why we believe this"
  }},
  "learning_notes": "What did we learn about agents or strategies?"
}}

REMEMBER: Failure is not the trigger - loss of progress is the signal.
"""


# ============================================================================
# UTILITY: Replanning Prompt
# ============================================================================

def create_replan_prompt(
    original_goal: str,
    failed_strategies: List[Strategy],
    context: PlannerContext
) -> str:
    """
    Generate prompt for replanning after strategy abandonment.
    
    This is NOT "generate a new plan" - it's "reframe the problem"
    given what we've learned.
    
    Args:
        original_goal: The user's goal
        failed_strategies: Strategies that didn't work
        context: What we've learned so far
        
    Returns:
        Formatted prompt for the LLM
    """
    failed_str = "\n\n".join([
        f"Strategy {i+1}: {s.hypothesis}\n"
        f"Assumptions: {s.assumptions}\n"
        f"Why it failed: Ran for {s.iterations} iterations, {s.health.value}\n"
        f"What we learned: {len(s.resolved_unknowns)} unknowns resolved"
        for i, s in enumerate(failed_strategies)
    ])
    
    facts_str = "\n".join([
        f"  - {k}: {v}" for k, v in context.known_facts.items()
    ]) if context.known_facts else "  (No facts established)"
    
    return f"""You are reframing a problem after previous strategies failed.

ORIGINAL GOAL:
{original_goal}

STRATEGIES THAT DIDN'T WORK:
{failed_str}

WHAT WE KNOW (facts established):
{facts_str}

YOUR TASK:
Propose a NEW way of thinking about this goal, informed by failures.

REFRAMING PRINCIPLES:
1. Don't just tweak - fundamentally reconsider
2. Question the goal itself if needed (maybe it's phrased wrong)
3. Use learned facts to inform new hypothesis
4. Avoid strategies we know don't work
5. Consider if the goal is actually achievable

OUTPUT REQUIREMENTS:
Provide structured JSON:
{{
  "reframed_goal": "Possibly refined goal statement",
  "new_hypothesis": "Fresh strategic approach",
  "why_different": "How this avoids previous failures",
  "new_assumptions": ["What we're assuming now"],
  "initial_sub_goals": ["Starting sub-goals for this framing"],
  "confidence_this_will_work": "high|medium|low",
  "alternative_framings": ["Other ways to think about this"],
  "goal_achievable": true/false,
  "reasoning": "Why this reframing makes sense"
}}

REMEMBER: Sometimes the goal itself needs refinement based on what we've learned.
"""