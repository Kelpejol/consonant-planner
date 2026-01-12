"""
Terra Planner Test Client

Simple client to test the planner gRPC service.
"""

import sys
import grpc
from src.proto.v1 import planner_pb2, planner_pb2_grpc


def test_planner(goal: str, context: dict = None):
    """
    Test the planner with a given goal.
    
    Args:
        goal: The planning goal
        context: Optional context dictionary
    """
    # Create channel
    channel = grpc.insecure_channel('localhost:50051')
    stub = planner_pb2_grpc.PlannerServiceStub(channel)
    
    # Create request
    request = planner_pb2.GeneratePlanRequest(
        goal=goal,
        workflow_id=f"test-{hash(goal)}",
        context=context or {}
    )
    
    print(f"🎯 Testing Planner with goal: {goal}")
    print(f"📝 Context: {context or {}}")
    print("\n" + "="*80 + "\n")
    
    try:
        # Call service
        response = stub.GeneratePlan(request)
        
        # Print results
        print("📊 PLANNING RESULTS\n")
        
        print(f"Generated {len(response.steps)} steps:\n")
        for i, step in enumerate(response.steps, 1):
            print(f"{i}. [{step.id}] {step.description}")
            print(f"   Agent: {step.agent_selector}")
            if step.dependencies:
                print(f"   Dependencies: {', '.join(step.dependencies)}")
            print()
        
        print("\n" + "="*80)
        print("\n🧠 REASONING TRACE\n")
        print(response.reasoning)
        print("\n" + "="*80)
        
        return response
        
    except grpc.RpcError as e:
        print(f"❌ RPC Error: {e.code()} - {e.details()}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Run test scenarios."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TERRA REFLECTIVE PLANNER TEST CLIENT                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test Scenario 1: System Debugging
    print("\n📍 TEST SCENARIO 1: System Debugging")
    print("-" * 80)
    test_planner(
        goal="Diagnose and fix payment processing failures in production",
        context={
            "service": "payment-processor",
            "environment": "production",
            "error_rate": "15%",
            "recent_changes": "database connection pool settings updated"
        }
    )
    
    # Test Scenario 2: Feature Development
    print("\n\n📍 TEST SCENARIO 2: Feature Development")
    print("-" * 80)
    test_planner(
        goal="Implement user authentication with OAuth2 and JWT tokens",
        context={
            "current_auth": "basic",
            "requirements": "OAuth2, JWT, refresh tokens",
            "tech_stack": "Python, FastAPI"
        }
    )
    
    # Test Scenario 3: Research Task
    print("\n\n📍 TEST SCENARIO 3: Research Task")
    print("-" * 80)
    test_planner(
        goal="Research and recommend the best approach for implementing real-time notifications",
        context={
            "scale": "10k concurrent users",
            "requirements": "low latency, reliable delivery",
            "current_infra": "AWS"
        }
    )
    
    print("\n\n✅ All tests complete!")


if __name__ == "__main__":
    main()