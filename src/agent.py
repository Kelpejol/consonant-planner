"""
Terra Reflective Planner Agent - FIXED VERSION

This module implements the core planning agent using LangGraph's state machine.

FIXES:
1. Corrected routing logic to match graph edges
2. Fixed LLM response parsing for Gemini's list format
3. Added missing context_dict initialization in decompose_goal
"""

import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from src.models import (
    PlanStep, SubGoal, Unknown, AgentCapability, AgentBehaviorHistory,
    PlannerContext, Strategy, ResultAnalysis, StrategyHealth, 
    ConfidenceLevel, PlanningPhase, ExecutionMode
)
from src.prompts import (
    create_understand_prompt, create_decompose_prompt, create_match_prompt,
    create_plan_prompt, create_analyze_prompt, create_decide_prompt,
    create_replan_prompt
)


# Configure structured logging
logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION (TypedDict for LangGraph performance)
# ============================================================================

class PlannerStateDict(TypedDict):
    """
    LangGraph state as TypedDict for performance.
    
    This mirrors PlannerState from models.py but uses TypedDict for LangGraph's
    internal optimization. The Pydantic model defines the contract and validation.
    """
    # Input
    goal: str
    workflow_id: str
    
    # Current phase
    current_phase: str  # PlanningPhase enum as string
    
    # Planning artifacts (stored as dicts for serialization)
    context: Dict[str, Any]  # PlannerContext serialized
    current_plan: List[Dict[str, Any]]  # List of PlanStep dicts
    
    # Iteration management
    iteration_count: int
    max_iterations: int
    
    # LLM interactions
    llm_messages: Annotated[List[BaseMessage], operator.add]  # Message accumulation
    
    # Decision tracking
    last_decision: str
    reasoning_trace: Annotated[List[str], operator.add]
    
    # Completion
    goal_satisfied: bool
    goal_impossible: bool
    final_reasoning: str


# ============================================================================
# REFLECTIVE PLANNER AGENT
# ============================================================================

class ReflectivePlannerAgent:
    """
    Production-grade reflective planning agent.
    
    This agent implements continuous, hypothesis-driven planning with:
    - Multi-phase cognitive loop (understand → decompose → match → plan → analyze → decide)
    - Strategic replanning when assumptions are invalidated
    - Agent behavior learning over time
    - Parallel execution planning
    - Unknown-driven reasoning
    
    The agent uses Google Gemini via langchain-google-genai for LLM reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        google_api_key: str = None
    ):
        """
        Initialize the reflective planner agent.
        
        Args:
            model_name: Gemini model to use
            temperature: LLM temperature (low for consistency)
            google_api_key: Google API key (if not in environment)
        """
        # Determine if we should run in mock mode
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.mock_mode = not api_key or api_key == "" or api_key.startswith("your-api-key")
        
        if self.mock_mode:
            logger.warning("Running in MOCK MODE (no valid API key found or placeholder used)")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    convert_system_message_to_human=True
                )
                logger.info(f"Initialized Gemini model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model ({e}). Falling back to MOCK mode.")
                self.mock_mode = True
                self.llm = None
        
        # Build the planning graph
        self.workflow = self._build_graph()
        logger.info("Planning workflow graph built successfully")
    
    # ========================================================================
    # NODE FUNCTIONS - Each implements a cognitive phase
    # ========================================================================
    
    def understand_goal(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        UNDERSTAND phase: Interpret and refine the goal.
        """
        logger.info(f"[{state['workflow_id']}] UNDERSTAND phase: Analyzing goal")
        
        context_dict = state.get('context', {})
        
        # Create prompt
        prompt = create_understand_prompt(
            goal=state['goal'],
            context=context_dict.get('known_facts', {})
        )
        
        # Query LLM (or mock for testing)
        if self.llm:
            try:
                messages = [
                    SystemMessage(content="You are a strategic planning expert."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                
                # Parse JSON response
                result = self._parse_llm_json(response.content)
                
                # Update context with understanding
                refined_goal = result.get('refined_goal', state['goal'])
                objectives = result.get('objectives', [state['goal']])
                unknowns_data = result.get('initial_unknowns', [])
                
                # Create Unknown objects
                unknowns = [
                    Unknown(
                        unknown_id=f"unk-understand-{i}",
                        description=unk,
                        importance=ConfidenceLevel.HIGH
                    ).model_dump()
                    for i, unk in enumerate(unknowns_data)
                ]
                
                # Update context
                updated_context = context_dict.copy()
                updated_context['known_facts'] = updated_context.get('known_facts', {})
                updated_context['known_facts']['refined_goal'] = refined_goal
                updated_context['known_facts']['objectives'] = objectives
                updated_context['remaining_unknowns'] = unknowns
                
                logger.info(f"[{state['workflow_id']}] Goal refined: {refined_goal}")
                
                return {
                    'context': updated_context,
                    'current_phase': PlanningPhase.DECOMPOSE.value,
                    'reasoning_trace': [f"UNDERSTAND: {result.get('confidence', 'unknown')} confidence in goal clarity"],
                    'llm_messages': [AIMessage(content=str(response.content))]
                }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] UNDERSTAND phase failed: {e}")
                return self._handle_phase_error(state, "understand", str(e))
        else:
            # Mock mode for testing
            return self._mock_understand(state)
    
    def decompose_goal(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        DECOMPOSE phase: Break goal into logical sub-goals.
        """
        logger.info(f"[{state['workflow_id']}] DECOMPOSE phase: Identifying sub-goals")
        
        # FIX: Initialize context_dict
        context_dict = state.get('context', {})
        known_facts = context_dict.get('known_facts', {})
        refined_goal = known_facts.get('refined_goal', state['goal'])
        objectives = known_facts.get('objectives', [state['goal']])
        
        # Get unknowns
        unknowns_data = context_dict.get('remaining_unknowns', [])
        unknowns = [Unknown(**u) if isinstance(u, dict) else u for u in unknowns_data]
        
        # Create prompt
        prompt = create_decompose_prompt(
            refined_goal=refined_goal,
            objectives=objectives,
            unknowns=unknowns,
            context=PlannerContext(**context_dict) if context_dict else PlannerContext()
        )
        
        if self.llm:
            try:
                messages = state['llm_messages'] + [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                result = self._parse_llm_json(response.content)
                
                # Create SubGoal objects
                sub_goals_data = result.get('sub_goals', [])
                sub_goals = []
                for sg_data in sub_goals_data:
                    # Create unknowns for this sub-goal
                    sg_unknowns = [
                        Unknown(
                            unknown_id=f"unk-{sg_data['id']}-{i}",
                            description=unk,
                            importance=ConfidenceLevel.HIGH,
                            related_sub_goal=sg_data['id']
                        )
                        for i, unk in enumerate(sg_data.get('unknowns_blocking', []))
                    ]
                    
                    sub_goal = SubGoal(
                        sub_goal_id=sg_data['id'],
                        description=sg_data['description'],
                        unknowns=sg_unknowns,
                        assumptions=sg_data.get('assumptions', []),
                        satisfaction_criteria=sg_data.get('satisfaction_criteria', ''),
                        confidence=ConfidenceLevel.LOW
                    )
                    sub_goals.append(sub_goal)
                
                # Update context
                updated_context = context_dict.copy()
                
                # Create initial strategy
                strategy = Strategy(
                    strategy_id=f"strategy-{state['iteration_count']}",
                    hypothesis=result.get('reasoning', 'Initial decomposition strategy'),
                    assumptions=[],
                    sub_goals=sub_goals,
                    health=StrategyHealth.VIABLE,
                    iterations=0
                )
                
                updated_context['active_strategy'] = strategy.model_dump()
                
                # Collect all unknowns from sub-goals
                all_unknowns = []
                for sg in sub_goals:
                    all_unknowns.extend([u.model_dump() for u in sg.unknowns])
                updated_context['remaining_unknowns'] = all_unknowns
                
                logger.info(f"[{state['workflow_id']}] Decomposed into {len(sub_goals)} sub-goals")
                
                return {
                    'context': updated_context,
                    'current_phase': PlanningPhase.MATCH.value,
                    'reasoning_trace': [f"DECOMPOSE: Identified {len(sub_goals)} sub-goals with {len(all_unknowns)} unknowns"],
                    'llm_messages': [AIMessage(content=str(response.content))]
                }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] DECOMPOSE phase failed: {e}")
                return self._handle_phase_error(state, "decompose", str(e))
        else:
            return self._mock_decompose(state)
    
    def match_capabilities(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        MATCH phase: Match unknowns to agent capabilities.
        """
        logger.info(f"[{state['workflow_id']}] MATCH phase: Matching agents to unknowns")
        
        context_dict = state.get('context', {})
        
        # Get active strategy
        strategy_data = context_dict.get('active_strategy', {})
        if not strategy_data or not isinstance(strategy_data, dict) or 'strategy_id' not in strategy_data:
            logger.warning(f"[{state['workflow_id']}] No active strategy in MATCH phase")
            return self._handle_phase_error(state, "match", "No active strategy found")
        
        strategy = Strategy(**strategy_data)
        sub_goals = strategy.sub_goals
        
        # Get unknowns
        unknowns_data = context_dict.get('remaining_unknowns', [])
        unknowns = [Unknown(**u) if isinstance(u, dict) else u for u in unknowns_data]
        
        # Get agent capabilities
        agent_capabilities = context_dict.get('agent_capabilities', {})
        agent_history = context_dict.get('agent_history', {})
        
        # Create prompt
        prompt = create_match_prompt(
            sub_goals=sub_goals,
            unknowns=unknowns,
            agent_capabilities=agent_capabilities,
            agent_history=agent_history,
            context=PlannerContext(**context_dict) if context_dict else PlannerContext()
        )
        
        if self.llm:
            try:
                messages = state['llm_messages'] + [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                result = self._parse_llm_json(response.content)
                
                # Store matches in context
                updated_context = context_dict.copy()
                updated_context['capability_matches'] = result.get('matches', [])
                updated_context['parallel_opportunities'] = result.get('parallel_opportunities', [])
                
                logger.info(f"[{state['workflow_id']}] Matched {len(result.get('matches', []))} unknowns to agents")
                
                return {
                    'context': updated_context,
                    'current_phase': PlanningPhase.PLAN.value,
                    'reasoning_trace': [f"MATCH: Created {len(result.get('matches', []))} capability matches"],
                    'llm_messages': [AIMessage(content=str(response.content))]
                }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] MATCH phase failed: {e}")
                return self._handle_phase_error(state, "match", str(e))
        else:
            return self._mock_match(state)
    
    def create_plan(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        PLAN phase: Structure concrete execution steps.
        """
        logger.info(f"[{state['workflow_id']}] PLAN phase: Structuring execution")
        
        context_dict = state.get('context', {})
        
        # Get strategy and matches
        strategy_data = context_dict.get('active_strategy', {})
        if not strategy_data or not isinstance(strategy_data, dict) or 'strategy_id' not in strategy_data:
            logger.warning(f"[{state['workflow_id']}] No active strategy in PLAN phase")
            return self._handle_phase_error(state, "plan", "No active strategy found")
            
        strategy = Strategy(**strategy_data)
        sub_goals = strategy.sub_goals
        
        matches = context_dict.get('capability_matches', [])
        parallel_groups = context_dict.get('parallel_opportunities', [])
        
        # Create prompt
        prompt = create_plan_prompt(
            sub_goals=sub_goals,
            matches=matches,
            parallel_groups=parallel_groups,
            context=PlannerContext(**context_dict)
        )
        
        if self.llm:
            try:
                messages = state['llm_messages'] + [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                result = self._parse_llm_json(response.content)
                
                # Create PlanStep objects
                steps_data = result.get('steps', [])
                plan_steps = []
                for step_data in steps_data:
                    step = PlanStep(
                        id=step_data['id'],
                        description=step_data['description'],
                        agent_selector=step_data['agent_selector'],
                        sub_goal_addressed=step_data.get('sub_goal_addressed', ''),
                        unknowns_targeted=step_data.get('unknowns_targeted', []),
                        dependencies=step_data.get('dependencies', []),
                        execution_mode=ExecutionMode(step_data.get('execution_mode', 'sequential')),
                        rationale=step_data.get('rationale', ''),
                        expected_outcome=step_data.get('expected_outcome', '')
                    )
                    plan_steps.append(step)
                
                logger.info(f"[{state['workflow_id']}] Created plan with {len(plan_steps)} steps")
                
                # Update iteration count
                new_iteration = state['iteration_count'] + 1
                
                return {
                    'current_plan': [s.model_dump() for s in plan_steps],
                    'current_phase': PlanningPhase.AWAIT.value,
                    'iteration_count': new_iteration,
                    'reasoning_trace': [f"PLAN: Generated {len(plan_steps)} execution steps"],
                    'llm_messages': [AIMessage(content=str(response.content))]
                }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] PLAN phase failed: {e}")
                return self._handle_phase_error(state, "plan", str(e))
        else:
            return self._mock_plan(state)
    
    def await_results(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        AWAIT phase: Wait for agent execution results.
        """
        logger.info(f"[{state['workflow_id']}] AWAIT phase: Simulating agent execution")
        
        current_plan = state.get('current_plan', [])
        
        # Simulate some results
        simulated_results = {}
        for step_dict in current_plan[:2]:  # Simulate first 2 steps
            step = PlanStep(**step_dict)
            simulated_results[step.id] = self._simulate_agent_result(step)
        
        # Store results in context
        context_dict = state.get('context', {})
        updated_context = context_dict.copy()
        updated_context['pending_results'] = simulated_results
        
        logger.info(f"[{state['workflow_id']}] Received {len(simulated_results)} simulated results")
        
        return {
            'context': updated_context,
            'current_phase': PlanningPhase.ANALYZE.value,
            'reasoning_trace': [f"AWAIT: Received results for {len(simulated_results)} steps"]
        }
    
    def analyze_results(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        ANALYZE phase: Interpret results and update context.
        """
        logger.info(f"[{state['workflow_id']}] ANALYZE phase: Interpreting results")
        
        context_dict = state.get('context', {})
        
        # Get results and plan
        step_results = context_dict.get('pending_results', {})
        current_plan = [PlanStep(**s) for s in state.get('current_plan', [])]
        
        # Get active strategy
        strategy_data = context_dict.get('active_strategy', {})
        if not strategy_data or not isinstance(strategy_data, dict) or 'strategy_id' not in strategy_data:
            logger.warning(f"[{state['workflow_id']}] No active strategy in ANALYZE phase")
            return self._handle_phase_error(state, "analyze", "No active strategy found")
            
        strategy = Strategy(**strategy_data)
        
        # Create prompt
        prompt = create_analyze_prompt(
            step_results=step_results,
            original_plan=current_plan,
            current_strategy=strategy,
            context=PlannerContext(**context_dict)
        )
        
        if self.llm:
            try:
                messages = state['llm_messages'] + [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                result = self._parse_llm_json(response.content)
                
                # Update context with analysis
                updated_context = context_dict.copy()
                
                # Update unknowns
                resolved_ids = [u['id'] for u in result.get('unknowns_resolved', [])]
                
                # Track strategy health
                strategic_assessment = result.get('strategic_assessment', {})
                if not strategic_assessment.get('continue_viable', True):
                    strategy.health = StrategyHealth.QUESTIONABLE
                
                strategy.iterations += 1
                updated_context['active_strategy'] = strategy.model_dump()
                
                # Store analysis
                updated_context['latest_analysis'] = result
                
                logger.info(f"[{state['workflow_id']}] Analysis: {len(resolved_ids)} unknowns resolved")
                
                return {
                    'context': updated_context,
                    'current_phase': PlanningPhase.DECIDE.value,
                    'reasoning_trace': [f"ANALYZE: Resolved {len(resolved_ids)} unknowns, strategy health: {strategy.health.value}"],
                    'llm_messages': [AIMessage(content=str(response.content))]
                }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] ANALYZE phase failed: {e}")
                return self._handle_phase_error(state, "analyze", str(e))
        else:
            return self._mock_analyze(state)
    
    def decide_next_move(self, state: PlannerStateDict) -> Dict[str, Any]:
        """
        DECIDE phase: Determine whether to continue, branch, or replan.
        """
        logger.info(f"[{state['workflow_id']}] DECIDE phase: Strategic decision")
        
        context_dict = state.get('context', {})
        
        # Get strategy and analysis
        strategy_data = context_dict.get('active_strategy', {})
        if not strategy_data or not isinstance(strategy_data, dict) or 'strategy_id' not in strategy_data:
            logger.warning(f"[{state['workflow_id']}] No active strategy in DECIDE phase")
            return self._handle_phase_error(state, "decide", "No active strategy found")
            
        strategy = Strategy(**strategy_data)
        
        latest_analysis = context_dict.get('latest_analysis', {})
        remaining_unknowns_data = context_dict.get('remaining_unknowns', [])
        remaining_unknowns = [Unknown(**u) for u in remaining_unknowns_data]
        
        # Check iteration limit
        if state['iteration_count'] >= state['max_iterations']:
            logger.warning(f"[{state['workflow_id']}] Max iterations reached")
            return {
                'goal_satisfied': False,
                'goal_impossible': True,
                'final_reasoning': "Maximum planning iterations exceeded",
                'current_phase': PlanningPhase.COMPLETE.value
            }
        
        # Create mock result analysis for decide prompt
        mock_analysis = ResultAnalysis(
            step_id="mock",
            agent_id="mock",
            raw_results="",
            confidence_in_results=ConfidenceLevel.MEDIUM,
            interpretation=latest_analysis.get('interpretation', 'Results analyzed'),
            unknowns_resolved=[],
            assumptions_invalidated=latest_analysis.get('assumptions_invalidated', []),
            suggests_replanning=not latest_analysis.get('strategic_assessment', {}).get('continue_viable', True)
        )
        
        # Create prompt
        prompt = create_decide_prompt(
            analysis=mock_analysis,
            current_strategy=strategy,
            remaining_unknowns=remaining_unknowns,
            iteration_count=state['iteration_count'],
            context=PlannerContext(**context_dict)
        )
        
        if self.llm:
            try:
                messages = state['llm_messages'] + [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                result = self._parse_llm_json(response.content)
                
                decision = result.get('decision', 'continue')
                goal_status = result.get('goal_status', {})
                
                logger.info(f"[{state['workflow_id']}] Decision: {decision}")
                
                # Handle decision
                if decision == 'complete' or goal_status.get('satisfied'):
                    return {
                        'goal_satisfied': True,
                        'final_reasoning': goal_status.get('explanation', 'Goal achieved'),
                        'current_phase': PlanningPhase.COMPLETE.value,
                        'last_decision': decision,
                        'reasoning_trace': [f"DECIDE: {decision} - {result.get('reasoning', '')}"],
                        'llm_messages': [AIMessage(content=str(response.content))]
                    }
                elif decision == 'replan' or goal_status.get('impossible'):
                    return {
                        'goal_impossible': goal_status.get('impossible', False),
                        'final_reasoning': result.get('reasoning', 'Replanning required'),
                        'current_phase': PlanningPhase.DECOMPOSE.value,
                        'last_decision': decision,
                        'reasoning_trace': [f"DECIDE: {decision} - replanning"],
                        'llm_messages': [AIMessage(content=str(response.content))]
                    }
                else:  # continue or branch
                    return {
                        'current_phase': PlanningPhase.DECOMPOSE.value,
                        'last_decision': decision,
                        'reasoning_trace': [f"DECIDE: {decision} - continuing"],
                        'llm_messages': [AIMessage(content=str(response.content))]
                    }
                
            except Exception as e:
                logger.error(f"[{state['workflow_id']}] DECIDE phase failed: {e}")
                return self._handle_phase_error(state, "decide", str(e))
        else:
            return self._mock_decide(state)
    
    # ========================================================================
    # ROUTING FUNCTIONS - FIXED
    # ========================================================================
    
    def route_from_phase(self, state: PlannerStateDict) -> str:
        """
        Route to next phase based on current phase in state.
        
        FIX: Returns the actual phase name that exists in the graph,
        not the enum value that might not match.
        """
        if state.get('goal_impossible') or state.get('goal_satisfied'):
            return "end"
            
        phase = state.get('current_phase', PlanningPhase.UNDERSTAND.value)
        
        if phase == PlanningPhase.COMPLETE.value:
            return "end"
        
        # Map phase enum values to node names
        phase_to_node = {
            PlanningPhase.UNDERSTAND.value: "understand",
            PlanningPhase.DECOMPOSE.value: "decompose",
            PlanningPhase.MATCH.value: "match",
            PlanningPhase.PLAN.value: "plan",
            PlanningPhase.AWAIT.value: "await",
            PlanningPhase.ANALYZE.value: "analyze",
            PlanningPhase.DECIDE.value: "decide",
            PlanningPhase.COMPLETE.value: "end"
        }
        
        return phase_to_node.get(phase, "end")
    
    def route_from_decide(self, state: PlannerStateDict) -> str:
        """
        Route after DECIDE phase based on decision.
        """
        if state.get('goal_satisfied') or state.get('goal_impossible'):
            return "end"
        
        phase = state.get('current_phase', '')
        if phase == PlanningPhase.COMPLETE.value:
            return "end"
        
        # Check iteration limit
        if state['iteration_count'] >= state['max_iterations']:
            return "end"
        
        # Continue looping - go back to understand for new iteration
        return "understand"
    
    # ========================================================================
    # GRAPH CONSTRUCTION - FIXED
    # ========================================================================
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine for the planning loop.
        
        FIX: Simplified conditional edges to only use valid destinations.
        """
        workflow = StateGraph(PlannerStateDict)
        
        # Add all cognitive phase nodes
        workflow.add_node("understand", self.understand_goal)
        workflow.add_node("decompose", self.decompose_goal)
        workflow.add_node("match", self.match_capabilities)
        workflow.add_node("plan", self.create_plan)
        workflow.add_node("await", self.await_results)
        workflow.add_node("analyze", self.analyze_results)
        workflow.add_node("decide", self.decide_next_move)
        
        # Set entry point
        workflow.set_entry_point("understand")
        
        # Simple linear edges for normal flow
        workflow.add_edge("understand", "decompose")
        workflow.add_edge("decompose", "match")
        workflow.add_edge("match", "plan")
        workflow.add_edge("plan", "await")
        workflow.add_edge("await", "analyze")
        workflow.add_edge("analyze", "decide")
        
        # Conditional routing from decide (loop or end)
        workflow.add_conditional_edges(
            "decide",
            self.route_from_decide,
            {
                "understand": "understand",  # Loop back
                "end": END
            }
        )
        
        return workflow.compile()
    
    # ========================================================================
    # UTILITY METHODS - FIXED
    # ========================================================================
    
    def _parse_llm_json(self, content: Any) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks and Gemini's list format.
        
        FIX: Properly handles Gemini's list response format.
        """
        # Handle list content (Gemini often returns list of content parts)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    # Gemini content parts have 'text' field
                    if "text" in part:
                        text_parts.append(part["text"])
                    # Also check for 'parts' field
                    elif "parts" in part:
                        for subpart in part["parts"]:
                            if isinstance(subpart, dict) and "text" in subpart:
                                text_parts.append(subpart["text"])
                            elif isinstance(subpart, str):
                                text_parts.append(subpart)
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
            
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
            
        # Remove markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}\nContent: {content[:500]}")
            raise
    
    def _handle_phase_error(
        self, 
        state: PlannerStateDict, 
        phase: str, 
        error: str
    ) -> Dict[str, Any]:
        """Handle errors during phase execution."""
        logger.error(f"Phase {phase} error: {error}")
        return {
            'goal_impossible': True,
            'final_reasoning': f"Planning failed in {phase} phase: {error}",
            'current_phase': PlanningPhase.COMPLETE.value
        }
    
    def _simulate_agent_result(self, step: PlanStep) -> str:
        """Simulate agent execution result."""
        return f"Simulated result for {step.description}: Analysis suggests {step.expected_outcome}"
    
    # ========================================================================
    # MOCK METHODS (for testing without LLM)
    # ========================================================================
    
    def _mock_understand(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock UNDERSTAND phase for testing."""
        context = state.get('context', {})
        context['known_facts'] = context.get('known_facts', {})
        context['known_facts']['refined_goal'] = state['goal']
        context['known_facts']['objectives'] = [state['goal']]
        context['remaining_unknowns'] = [
            Unknown(
                unknown_id="unk-1",
                description="Need to determine failure cause",
                importance=ConfidenceLevel.HIGH
            ).model_dump()
        ]
        
        return {
            'context': context,
            'current_phase': PlanningPhase.DECOMPOSE.value,
            'reasoning_trace': ["UNDERSTAND: Mock mode"]
        }
    
    def _mock_decompose(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock DECOMPOSE phase for testing."""
        context = state.get('context', {})
        
        sub_goal = SubGoal(
            sub_goal_id="sg-1",
            description="Identify failure category",
            unknowns=[Unknown(
                unknown_id="unk-sg1-1",
                description="What type of failure is occurring?",
                importance=ConfidenceLevel.HIGH,
                related_sub_goal="sg-1"
            )],
            assumptions=["Failures are detectable in logs"],
            satisfaction_criteria="Failure type is known with high confidence"
        )
        
        strategy = Strategy(
            strategy_id="strategy-1",
            hypothesis="Failures are infrastructure-related",
            sub_goals=[sub_goal],
            health=StrategyHealth.VIABLE
        )
        
        context['active_strategy'] = strategy.model_dump()
        context['remaining_unknowns'] = [u.model_dump() for u in sub_goal.unknowns]
        
        return {
            'context': context,
            'current_phase': PlanningPhase.MATCH.value,
            'reasoning_trace': ["DECOMPOSE: Mock mode"]
        }
    
    def _mock_match(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock MATCH phase for testing."""
        context = state.get('context', {})
        context['capability_matches'] = [
            {
                'unknown_id': 'unk-sg1-1',
                'primary_agent': 'log_analyzer',
                'rationale': 'Mock match',
                'execution_strategy': 'parallel'
            }
        ]
        
        return {
            'context': context,
            'current_phase': PlanningPhase.PLAN.value,
            'reasoning_trace': ["MATCH: Mock mode"]
        }
    
    def _mock_plan(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock PLAN phase for testing."""
        step = PlanStep(
            id="step-1",
            description="Analyze logs for failure patterns",
            agent_selector="log_analyzer",
            sub_goal_addressed="sg-1",
            unknowns_targeted=["unk-sg1-1"],
            dependencies=[],
            execution_mode=ExecutionMode.PARALLEL,
            rationale="Mock rationale",
            expected_outcome="Failure type identified"
        )
        
        return {
            'current_plan': [step.model_dump()],
            'current_phase': PlanningPhase.AWAIT.value,
            'iteration_count': state['iteration_count'] + 1,
            'reasoning_trace': ["PLAN: Mock mode"]
        }
    
    def _mock_analyze(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock ANALYZE phase for testing."""
        context = state.get('context', {})
        context['latest_analysis'] = {
            'unknowns_resolved': [{'id': 'unk-sg1-1', 'resolution': 'Mock resolution'}],
            'strategic_assessment': {'continue_viable': True}
        }
        
        return {
            'context': context,
            'current_phase': PlanningPhase.DECIDE.value,
            'reasoning_trace': ["ANALYZE: Mock mode"]
        }
    
    def _mock_decide(self, state: PlannerStateDict) -> Dict[str, Any]:
        """Mock DECIDE phase for testing."""
        # After 2 iterations in mock mode, complete
        if state['iteration_count'] >= 2:
            return {
                'goal_satisfied': True,
                'final_reasoning': "Mock completion after 2 iterations",
                'current_phase': PlanningPhase.COMPLETE.value
            }
        
        return {
            'current_phase': PlanningPhase.DECOMPOSE.value,
            'reasoning_trace': ["DECIDE: Mock continue"]
        }