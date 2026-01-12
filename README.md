# Terra Reflective Planner Service

A production-grade, oracle-level AI planning system built with LangGraph and gRPC that implements continuous, reflective planning with agent capability matching and behavioral learning.

## 🎯 System Overview

The Terra Planner is **NOT** an executor - it is a **thinking machine** that:

- **Decides** what should be done next given goals, context, and results
- **Reasons** about dependencies, unknowns, and information gaps
- **Learns** from agent behavior over time
- **Replans** strategically when assumptions are invalidated
- **Parallelizes** independent sub-goals intelligently

### Key Principles

1. **Planner vs Plan**: The planner is long-lived and stateful; plans are ephemeral snapshots
2. **Continuous Planning**: Not "plan once then execute" - continuous reasoning loop
3. **Unknown-Driven**: Focuses on "what do I not know?" rather than "what steps to run?"
4. **Agent Learning**: Builds semantic understanding of agent capabilities over time
5. **Strategy-Based**: Plans are hypotheses that can be abandoned or branched

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     gRPC Service Layer                      │
│  (PlannerService - External Interface)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Reflective Planner Engine                      │
│  (LangGraph State Machine - Cognitive Loop)                 │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌───────────┐            │
│  │ UNDERSTAND │→ │ DECOMPOSE  │→ │   MATCH   │            │
│  └────────────┘  └────────────┘  └─────┬─────┘            │
│                                         │                   │
│  ┌────────────┐  ┌────────────┐  ┌─────▼─────┐            │
│  │  ANALYZE   │← │   AWAIT    │← │   PLAN    │            │
│  └─────┬──────┘  └────────────┘  └───────────┘            │
│        │                                                    │
│  ┌─────▼──────┐                                            │
│  │   DECIDE   │ (continue/branch/replan)                   │
│  └─────┬──────┘                                            │
│        └──────────────────┐                                │
│                           ↺ (loop until goal satisfied)    │
└────────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  LLM Integration Layer                      │
│        (Google Gemini via langchain-google-genai)           │
└─────────────────────────────────────────────────────────────┘
```

### State Machine Phases

1. **UNDERSTAND**: Goal interpretation and constraint identification
2. **DECOMPOSE**: Break goal into logical sub-goals  
3. **MATCH**: Find agent capabilities that address unknowns
4. **PLAN**: Structure execution (parallel/sequential/conditional)
5. **AWAIT**: Wait for results (in production this would trigger agent execution)
6. **ANALYZE**: Interpret results and update context
7. **DECIDE**: Continue, branch hypotheses, or replan

## 📦 Installation

### Prerequisites

- Python 3.10+ (tested on 3.12.3)
- Google API key for Gemini

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd terra-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Generate Proto Files

```bash
# Generate Python gRPC code from proto definitions
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src/proto/v1 \
    --grpc_python_out=./src/proto/v1 \
    ./proto/v1/planner.proto
```

## 🚀 Usage

### Start gRPC Server

```bash
# Development mode with logging
python -m src.server

# Production mode
python -m src.server --log-level INFO
```

The server will start on `localhost:50051` by default.

### Example gRPC Client

```python
import grpc
from src.proto.v1 import planner_pb2, planner_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = planner_pb2_grpc.PlannerServiceStub(channel)

# Create planning request
request = planner_pb2.GeneratePlanRequest(
    goal="Diagnose and fix payment processing failures",
    workflow_id="workflow-001",
    context={
        "service": "payment-processor",
        "environment": "production",
        "error_rate": "15%"
    }
)

# Generate plan
response = stub.GeneratePlan(request)

# Process plan steps
for step in response.steps:
    print(f"Step {step.id}: {step.description}")
    print(f"  Agent Type: {step.agent_selector}")
    print(f"  Dependencies: {step.dependencies}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Planning Process

### Example Planning Flow

**Goal**: "Diagnose why payments are failing and fix it"

**Iteration 1 - Initial Understanding**:
```
UNDERSTAND → Goal reframed as:
  - Objective A: Identify failure cause
  - Objective B: Apply corrective action
  - Unknowns: error type, scope, recent changes

DECOMPOSE → Sub-goals identified:
  - Determine failure category
  - Identify scope of impact
  - Find recent changes
  - Validate fix

MATCH → Agents selected:
  - Log analyzer (forensics capability)
  - Metrics analyzer (observability capability)
  - Deployment tracker (change history capability)

PLAN → Execution structure:
  - Parallel: [log_analysis, metrics_analysis]
  - Sequential: deployment_check depends on failure_category
```

**Iteration 2 - After Results**:
```
ANALYZE → Results processed:
  - Logs indicate database timeout
  - Metrics show connection pool exhaustion
  - Recent deploy changed connection settings

DECIDE → Strategy remains valid, continue with:
  - Hypothesis: Connection pool misconfiguration
  - Next steps: Verify settings, propose fix
```

**Iteration 3 - Replanning**:
```
ANALYZE → Assumption invalidated:
  - Settings are correct
  - Issue predates recent deploy

DECIDE → Strategy abandoned, new framing:
  - From "config issue" to "resource exhaustion"
  - New sub-goals: Check resource limits, analyze load patterns
```

## 🎯 Key Features

### 1. Reflective Planning Loop

- Continuous evaluation of strategy viability
- Automatic replanning when assumptions are invalidated
- Confidence tracking for all conclusions

### 2. Agent Behavior Learning

- Semantic model of agent capabilities (not just static tags)
- Historical pattern recognition (e.g., "Agent X often misses edge cases")
- Trust weighting based on past performance
- Informed capability matching

### 3. Parallel Execution Planning

- Identifies independent sub-goals that can run concurrently
- Explicit dependency management
- Early result invalidation detection

### 4. Unknown-Driven Reasoning

- Focuses on information gaps rather than procedural steps
- Tracks resolved vs remaining unknowns
- Measures uncertainty reduction

### 5. Strategy Management

- Hypothesis-based planning (strategies are assumptions)
- Branching for multiple plausible explanations
- Aggressive pruning of unproductive paths

## 🔧 Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=AIza...

# Optional
LOG_LEVEL=INFO                    # Logging level (DEBUG, INFO, WARNING, ERROR)
GRPC_PORT=50051                   # gRPC server port
MAX_PLANNING_ITERATIONS=10        # Max iterations before forced termination
PLANNER_MODEL=gemini-1.5-flash    # Gemini model to use
PLANNING_TEMPERATURE=0.1          # LLM temperature for planning
```

### Planner Configuration

Edit `src/config.py` to adjust:

- Planning prompts
- Cognitive phase transitions
- Result analysis thresholds
- Replanning triggers

## 📝 Development

### Project Structure

```
terra-planner/
├── proto/
│   └── v1/
│       └── planner.proto          # gRPC service definition
├── src/
│   ├── proto/
│   │   └── v1/
│   │       ├── planner_pb2.py     # Generated protobuf code
│   │       └── planner_pb2_grpc.py
│   ├── __init__.py
│   ├── agent.py                   # Core planning agent logic
│   ├── config.py                  # Configuration management
│   ├── models.py                  # Pydantic data models
│   ├── prompts.py                 # LLM prompts for each phase
│   ├── server.py                  # gRPC server implementation
│   └── utils.py                   # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_server.py
│   └── test_integration.py
├── requirements.txt
├── .env.example
└── README.md
```

### Adding New Cognitive Phases

1. Define phase in `src/agent.py` as a node function
2. Add phase-specific prompts in `src/prompts.py`
3. Update state transitions in the graph builder
4. Add tests in `tests/test_agent.py`

## 🛡️ Production Considerations

### Error Handling

- Comprehensive try-catch blocks with structured logging
- Graceful degradation when LLM calls fail
- Timeout protection for infinite planning loops
- Validation of all state transitions

### Performance

- Connection pooling for gRPC
- Async LLM calls where possible
- Result caching for repeated queries
- Efficient state serialization

### Monitoring

- Structured logging with context
- Planning iteration metrics
- Agent selection statistics
- Strategy abandonment tracking

### Security

- API key management via environment variables
- Input validation at gRPC boundary
- No sensitive data in logs
- Rate limiting (add as needed)

## 📚 Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API](https://ai.google.dev/)
- [gRPC Python Guide](https://grpc.io/docs/languages/python/)
- [Planning System Design Document](docs/DESIGN.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built on LangGraph by LangChain
- Powered by Google Gemini
- Inspired by scientific method and hypothesis-driven research