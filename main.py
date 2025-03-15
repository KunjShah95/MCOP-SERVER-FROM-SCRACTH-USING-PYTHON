import json
import uuid
import datetime
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Callable
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# --- Core MCP Classes ---
class Tool:
    def __init__(self, name, description, function: Callable):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.function = function

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }

class MCP:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.context_id = str(uuid.uuid4())

    def register_tool(self, tool):
        self.tools[tool.id] = tool

    def tools_list(self):
        tool_list = [tool.to_dict() for tool in self.tools.values()]
        return self._json_rpc_response("tools_list", tool_list)

    def tool_call(self, tool_id, input_data):
        tool = self.tools.get(tool_id)
        if not tool:
            return self._json_rpc_error("tool_not_found", f"Tool with ID {tool_id} not found.")

        try:
            # Simulate streaming result (e.g., for a long-running task)
            def stream_result():
                result = tool.function(**input_data)
                yield f"data: {result}\n\n"  # SSE format

            return StreamingResponse(stream_result(), media_type="text/event-stream")

        except Exception as e:
            return self._json_rpc_error("tool_execution_error", str(e))

    def _json_rpc_response(self, method, result):
        return {
            "id": self.context_id,
            "jsonrpc": "2.0",
            "result": result,
            "method": method,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _json_rpc_error(self, code, message):
        return {
            "id": self.context_id,
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "method": None,
            "timestamp": datetime.datetime.now().isoformat()
        }

# --- FastAPI Application ---
app = FastAPI()

security = HTTPBasic()

# Initialize MCP instance
mcp = MCP()

# Predefined Tools (No dynamic registration anymore)
def get_current_weather(location: str) -> str:
    return f"The weather in {location} is sunny with a temperature of 25°C."

def calculate_sum(a: int, b: int) -> int:
    return a + b

weather_tool = Tool(name="get_current_weather", description="Gets the current weather.", function=get_current_weather)
sum_tool = Tool(name="calculate_sum", description="Calculates the sum of two numbers.", function=calculate_sum)
mcp.register_tool(weather_tool)
mcp.register_tool(sum_tool)

# --- API Endpoints ---
@app.get("/tools")
async def list_tools(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "user" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return mcp.tools_list()

@app.post("/tool_call/{tool_id}")
async def call_tool(tool_id: str, data: Dict[str, Any], credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "user" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return mcp.tool_call(tool_id, data)

# Removed /register_tool endpoint.  Tools are now predefined.

# --- LLM Integration Simulation ---
@app.get("/llm_request")
async def llm_request(query: str):
    """Simulates an LLM requesting tools."""
    if "weather" in query.lower():
        location = query.split("in ")[-1].strip()
        tool_call_response = await call_tool("get_current_weather", {"location": location})
        return {"response": tool_call_response}
    elif "sum" in query.lower():
        try:
            parts = query.split("sum of ")
            numbers = parts[1].split(" and ")
            a = int(numbers[0])
            b = int(numbers[1])
            tool_call_response = await call_tool("calculate_sum", {"a": a, "b": b})
            return {"response": tool_call_response}
        except Exception as e:
            return {"error": "Invalid query format"}
    else:
        return {"response": "I don't know how to handle that query."}

# --- Run the app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
