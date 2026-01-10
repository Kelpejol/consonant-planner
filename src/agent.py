from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ============================================================================
# STATE DEFINITION
# ============================================================================

class PlanStep(BaseModel):
    id: str = Field(description="Unique identifier for the step")
    description: str = Field(description="Detailed description of the step")
    agent_selector: str = Field(description="Type of agent required (e.g., coder, researcher)")
    dependencies: List[str] = Field(default_factory=list, description="IDs of steps this depends on")

class PlannerState(TypedDict):
    goal: str
    context: dict
    plan: List[PlanStep]
    critique: str
    revision_count: int
    messages: Annotated[List[BaseMessage], operator.add]

# ============================================================================
# NODES
# ============================================================================

class ReflectivePlannerAgent:
    def __init__(self, model_name="gpt-4o"):
        # self.model = ChatOpenAI(model=model_name)
        # Mocking for now to ensure structural correctness without API keys
        self.mock_mode = True

    def create_plan(self, state: PlannerState):
        """Generates initial plan based on goal."""
        print(f"--- GENERATING PLAN FOR: {state['goal']} ---")
        
        # Stub logic for verification
        plan = [
            PlanStep(
                id="step-1", 
                description="Analyze requirements", 
                agent_selector="researcher",
                dependencies=[]
            ),
            PlanStep(
                id="step-2", 
                description="Implement core logic", 
                agent_selector="coder", 
                dependencies=["step-1"]
            ),
             PlanStep(
                id="step-3", 
                description="Verify implementation", 
                agent_selector="tester", 
                dependencies=["step-2"]
            )
        ]
        
        return {"plan": plan, "revision_count": state.get("revision_count", 0) + 1}

    def reflect(self, state: PlannerState):
        """Critiques the plan."""
        print("--- REFLECTING ON PLAN ---")
        
        # Stub logic
        # In real impl, checking if plan is valid
        if state["revision_count"] > 1:
            return {"critique": "looks_good"}
        
        return {"critique": "needs_refinement"}

    def route_step(self, state: PlannerState):
        """Decides next step based on critique."""
        critique = state.get("critique")
        if critique == "looks_good" or state["revision_count"] > 3:
            return "formatted"
        return "planner"

    def build_graph(self):
        workflow = StateGraph(PlannerState)
        
        workflow.add_node("planner", self.create_plan)
        workflow.add_node("reflector", self.reflect)
        
        workflow.set_entry_point("planner")
        
        workflow.add_edge("planner", "reflector")
        
        workflow.add_conditional_edges(
            "reflector",
            self.route_step,
            {
                "planner": "planner",
                "formatted": END
            }
        )
        
        return workflow.compile()
