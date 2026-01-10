from concurrent import futures
import logging
import grpc
from proto.v1 import planner_pb2
from proto.v1 import planner_pb2_grpc
from agent import ReflectivePlannerAgent, PlannerState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("planner_server")

class PlannerService(planner_pb2_grpc.PlannerServiceServicer):
    def __init__(self):
        self.agent = ReflectivePlannerAgent()
        self.workflow = self.agent.build_graph()

    def GeneratePlan(self, request, context):
        logger.info(f"Received planning request for workflow: {request.workflow_id}")
        logger.info(f"Goal: {request.goal}")
        
        # Invoke LangGraph agent
        initial_state: PlannerState = {
            "goal": request.goal,
            "context": dict(request.context),
            "plan": [],
            "critique": "",
            "revision_count": 0,
            "messages": []
        }
        
        # Run the graph
        # For simple cases we can just invoke the creation logic directly or run the graph
        # Since invoke returns the final state
        final_state = self.workflow.invoke(initial_state)
        
        # Extract plan
        plan_steps = final_state.get("plan", [])
        
        # Convert to gRPC response
        response = planner_pb2.GeneratePlanResponse()
        response.reasoning = final_state.get("critique", "Generated via reflective agent")
        
        for step in plan_steps:
            grpc_step = response.steps.add()
            grpc_step.id = step.id
            grpc_step.description = step.description
            grpc_step.agent_selector = step.agent_selector
            grpc_step.dependencies.extend(step.dependencies)
            
        logger.info(f"Returning plan with {len(response.steps)} steps")
        return response

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    planner_pb2_grpc.add_PlannerServiceServicer_to_server(PlannerService(), server)
    server.add_insecure_port("[::]:" + port)
    logger.info("Planner Service started, listening on " + port)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
