"""
Terra Planner gRPC Server

Production-grade gRPC service for the reflective planner.

This server exposes the planner via gRPC, handling:
- Request validation
- Workflow orchestration
- Error handling and recovery
- Structured logging
- Graceful shutdown

The server is designed to be horizontally scalable and production-ready.
"""

import os
import sys
import signal
import logging
import structlog
from concurrent import futures
from typing import Dict, Any
from datetime import datetime

try:
    import grpc
    from src.proto.v1 import planner_pb2, planner_pb2_grpc
except ImportError:
    print("ERROR: gRPC packages not installed. Run: pip install grpcio grpcio-tools")
    sys.exit(1)

from src.agent import ReflectivePlannerAgent, PlannerStateDict
from src.models import PlanStep, PlannerContext


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ============================================================================
# GRPC SERVICE IMPLEMENTATION
# ============================================================================

class PlannerService(planner_pb2_grpc.PlannerServiceServicer):
    """
    gRPC service implementation for the Terra Planner.
    
    This service receives planning requests, invokes the reflective planner agent,
    and returns structured plans with full reasoning traces.
    
    Thread-safe and designed for concurrent request handling.
    """
    
    def __init__(self, google_api_key: str = None):
        """
        Initialize the planner service.
        
        Args:
            google_api_key: Google API key for Gemini (if not in env)
        """
        # Initialize the planner agent
        self.agent = ReflectivePlannerAgent(
            model_name=os.getenv("PLANNER_MODEL", "gemini-1.5-flash"),
            temperature=float(os.getenv("PLANNING_TEMPERATURE", "0.1")),
            google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY")
        )
        
        logger.info("planner_service_initialized", 
                   model=os.getenv("PLANNER_MODEL", "gemini-1.5-flash"))
    
    def GeneratePlan(
        self, 
        request: planner_pb2.GeneratePlanRequest, 
        context: grpc.ServicerContext
    ) -> planner_pb2.GeneratePlanResponse:
        """
        Generate a plan for the given goal.
        
        This is the main RPC method that:
        1. Validates the request
        2. Invokes the planning agent
        3. Converts results to gRPC response
        4. Handles errors gracefully
        
        Args:
            request: The planning request containing goal, workflow_id, and context
            context: gRPC context for the request
            
        Returns:
            GeneratePlanResponse with plan steps and reasoning
        """
        start_time = datetime.utcnow()
        workflow_id = request.workflow_id or f"workflow-{datetime.utcnow().timestamp()}"
        
        logger.info(
            "plan_request_received",
            workflow_id=workflow_id,
            goal=request.goal,
            context_keys=list(request.context.keys()) if request.context else []
        )
        
        # Validate request
        try:
            self._validate_request(request)
        except ValueError as e:
            logger.error("invalid_request", workflow_id=workflow_id, error=str(e))
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return planner_pb2.GeneratePlanResponse()
        
        # Build initial state for the planner
        initial_state: PlannerStateDict = {
            'goal': request.goal,
            'workflow_id': workflow_id,
            'current_phase': 'understand',
            'context': {
                'known_facts': dict(request.context) if request.context else {},
                'resolved_unknowns': {},
                'remaining_unknowns': [],
                'agent_capabilities': {},
                'agent_history': {},
                'completed_steps': [],
                'result_history': []
            },
            'current_plan': [],
            'iteration_count': 0,
            'max_iterations': int(os.getenv("MAX_PLANNING_ITERATIONS", "10")),
            'llm_messages': [],
            'last_decision': '',
            'reasoning_trace': [],
            'goal_satisfied': False,
            'goal_impossible': False,
            'final_reasoning': ''
        }
        
        try:
            # Invoke the planning workflow
            logger.info("invoking_planner_workflow", workflow_id=workflow_id)
            
            final_state = self.agent.workflow.invoke(initial_state)
            
            # Extract results
            plan_steps = final_state.get('current_plan', [])
            reasoning_trace = final_state.get('reasoning_trace', [])
            goal_satisfied = final_state.get('goal_satisfied', False)
            goal_impossible = final_state.get('goal_impossible', False)
            final_reasoning = final_state.get('final_reasoning', '')
            iterations = final_state.get('iteration_count', 0)
            
            # Log completion
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "plan_generation_complete",
                workflow_id=workflow_id,
                goal_satisfied=goal_satisfied,
                goal_impossible=goal_impossible,
                iterations=iterations,
                steps_generated=len(plan_steps),
                duration_seconds=duration
            )
            
            # Build gRPC response
            response = planner_pb2.GeneratePlanResponse()
            
            # Add plan steps
            for step_dict in plan_steps:
                try:
                    step = PlanStep(**step_dict)
                    grpc_step = response.steps.add()
                    grpc_step.id = step.id
                    grpc_step.description = step.description
                    grpc_step.agent_selector = step.agent_selector
                    grpc_step.dependencies.extend(step.dependencies)
                except Exception as e:
                    logger.warning("step_conversion_failed", 
                                 workflow_id=workflow_id, 
                                 error=str(e))
                    continue
            
            # Build reasoning summary
            reasoning_summary = self._build_reasoning_summary(
                reasoning_trace=reasoning_trace,
                goal_satisfied=goal_satisfied,
                goal_impossible=goal_impossible,
                final_reasoning=final_reasoning,
                iterations=iterations
            )
            response.reasoning = reasoning_summary
            
            return response
            
        except Exception as e:
            logger.error(
                "plan_generation_failed",
                workflow_id=workflow_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            
            # Return error response
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Planning failed: {str(e)}"
            )
            return planner_pb2.GeneratePlanResponse()
    
    def _validate_request(self, request: planner_pb2.GeneratePlanRequest) -> None:
        """
        Validate the planning request.
        
        Args:
            request: The gRPC request to validate
            
        Raises:
            ValueError: If request is invalid
        """
        if not request.goal or not request.goal.strip():
            raise ValueError("goal must be a non-empty string")
        
        if len(request.goal) > 10000:
            raise ValueError("goal must be less than 10000 characters")
    
    def _build_reasoning_summary(
        self,
        reasoning_trace: list,
        goal_satisfied: bool,
        goal_impossible: bool,
        final_reasoning: str,
        iterations: int
    ) -> str:
        """
        Build a human-readable reasoning summary.
        
        Args:
            reasoning_trace: List of reasoning steps
            goal_satisfied: Whether goal was satisfied
            goal_impossible: Whether goal was deemed impossible
            final_reasoning: Final reasoning explanation
            iterations: Number of planning iterations
            
        Returns:
            Formatted reasoning summary
        """
        lines = []
        
        # Status
        if goal_satisfied:
            lines.append("✅ Goal Satisfied")
        elif goal_impossible:
            lines.append("❌ Goal Deemed Impossible")
        else:
            lines.append("⚠️  Planning Incomplete")
        
        lines.append(f"\nIterations: {iterations}")
        lines.append("")
        
        # Reasoning trace
        if reasoning_trace:
            lines.append("Planning Process:")
            for i, trace in enumerate(reasoning_trace[-10:], 1):  # Last 10 steps
                lines.append(f"  {i}. {trace}")
            lines.append("")
        
        # Final reasoning
        if final_reasoning:
            lines.append("Final Assessment:")
            lines.append(f"  {final_reasoning}")
        
        return "\n".join(lines)


# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

def serve(port: int = 50051, max_workers: int = 10):
    """
    Start the gRPC server.
    
    Args:
        port: Port to listen on
        max_workers: Maximum concurrent workers
    """
    # Create server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # Add service
    planner_pb2_grpc.add_PlannerServiceServicer_to_server(
        PlannerService(),
        server
    )
    
    # Bind to port
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info("server_starting", port=port, max_workers=max_workers)
    
    # Start server
    server.start()
    logger.info("server_started", port=port)
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        logger.info("gracefully_stopping_server")
        server.stop(grace=10)
        logger.info("server_stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for termination
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        server.stop(grace=10)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the server."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(message)s',
    )
    
    # Get configuration
    port = int(os.getenv("GRPC_PORT", "50051"))
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your-api-key-here":
        logger.warning(
            "google_api_key_not_found",
            message="GOOGLE_API_KEY not set or placeholder - running in MOCK MODE"
        )
    
    # Start server
    serve(port=port, max_workers=max_workers)


if __name__ == "__main__":
    main()