---
theme: neversink
title: Building Reliable Agentic AI Systems
info: |
  ## Building Reliable AI Systems with Generative Models

  Graduate-level lecture on practical insights for building reliable
  agentic systems using LLMs. Focus on simplicity, patterns, and
  production-ready approaches.

  UCF MSDA - Final Semester
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Building Reliable Agentic AI Systems

From Theory to Production

<div class="text-gray-500 mt-8">
Spencer Lyon <br />
CAP-6942: Capstone<br/>
University of Central Florida
</div>

<div @click="$slidev.nav.next" class="mt-12 py-1 cursor-pointer" hover:bg="white op-10">
  Press Space for next page <carbon:arrow-right />
</div>

<!--
Welcome to our lecture on building reliable agentic AI systems. Today we'll focus on practical insights from real production systems, not just theoretical concepts.
-->

---

# What is an Agent?

<div class="text-4xl mt-8">
  <span class="text-blue-500">LLM</span> +
  <span class="text-green-500">Tools</span> +
  <span class="text-purple-500">Loop</span>
</div>

<div class="mt-8 text-xl text-gray-400">
Not chatbots, but autonomous systems<br/>
Simple definition beats complex abstractions
</div>

<!--
An agent is fundamentally just three things combined: an LLM for reasoning, tools to interact with the world, and a loop to keep it running. That's it. Don't overcomplicate it.
-->

---

# The Uncomfortable Truth About Production

<div class="grid grid-cols-2 gap-8 mt-8">
  <div>
    <h3 class="text-blue-500 mb-4">What You See in Tutorials</h3>

    - Complex frameworks everywhere
    - "Autonomous" agents doing everything
    - LangChain, CrewAI, AutoGPT
    - Multi-agent swarms
  </div>

  <div>
    <h3 class="text-red-500 mb-4">What's Actually in Production</h3>

    - Custom building blocks
    - Mostly deterministic code
    - Strategic LLM calls only
    - Simple, debuggable systems
  </div>
</div>

<div class="mt-12 p-4 bg-yellow-500 bg-opacity-10 rounded">
💡 **Key Insight**: Most successful AI applications are built with custom building blocks, not frameworks
</div>

<!--
There's a huge gap between what you see in demos and what actually works in production. The frameworks aren't being used. Teams build custom solutions with simple, composable patterns.
-->

---

# LLM Calls: Expensive and Dangerous

```python
# The most expensive operation in modern software
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    temperature=0.7  # Probabilistic outputs!
)
```

<div class="grid grid-cols-3 gap-6 mt-8">
  <div class="p-4 bg-red-500 bg-opacity-10 rounded">
    <h4 class="text-red-500">💸 Cost</h4>
    <p class="text-sm mt-2">Every call costs money</p>
  </div>

  <div class="p-4 bg-orange-500 bg-opacity-10 rounded">
    <h4 class="text-orange-500">⏱️ Latency</h4>
    <p class="text-sm mt-2">Seconds, not milliseconds</p>
  </div>

  <div class="p-4 bg-yellow-500 bg-opacity-10 rounded">
    <h4 class="text-yellow-500">🎲 Non-deterministic</h4>
    <p class="text-sm mt-2">Different outputs each time</p>
  </div>
</div>

<div class="mt-8 text-xl text-center text-gray-400">
Only use when deterministic code fails
</div>

<!--
LLM API calls are the most expensive and dangerous operation in modern software. They cost money, they're slow, and they're unpredictable. Use them strategically, not everywhere.
-->

---
layout: two-cols-title
---

:: title ::

# Workflows vs Agents

::left::

<div class="mt-8">
  <h3 class="text-blue-500 mb-4">Workflows</h3>

- **Predefined code paths**
- You control the flow
- LLM makes decisions within constraints
- Predictable and debuggable

```python
# Workflow example
if classify_intent(msg) == "complaint":
    response = handle_complaint(msg)
else:
    response = answer_question(msg)
```

</div>

::right::

<div class="mt-8">
  <h3 class="text-purple-500 mb-4">Agents</h3>

- **LLM controls the process**
- Dynamic decision making
- Can call tools autonomously
- Flexible but unpredictable

  ```python
  # Agent example
  while not done:
      action = llm.decide_action()
      result = execute(action)
      done = llm.is_complete(result)
  ```

</div>

<div class="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-center">
  <div class="text-xl text-gray-400">Start with workflows, evolve to agents when needed</div>
</div>

<!--
Understanding this distinction is crucial. Workflows are predictable - you define the path. Agents are autonomous - the LLM decides. Most production systems start as workflows.
-->

---

# Context Engineering > Prompt Engineering

<div class="grid grid-cols-2 gap-8 mt-8">
  <div>
    <h3 class="text-gray-500 mb-4">Old Way: Prompt Engineering</h3>

```python
prompt = """You are an expert assistant.
Please help the user with their request.
Be concise and helpful."""

response = llm(prompt + user_input)
```

Static, limited, hope for the best
  </div>

  <div>
    <h3 class="text-green-500 mb-4">New Way: Context Engineering</h3>

```python
context = {
    "user_history": get_user_history(),
    "relevant_docs": search(user_input),
    "current_time": datetime.now(),
    "user_preferences": load_preferences()
}

response = llm(build_prompt(context))
```

Dynamic, comprehensive, reliable
  </div>
</div>

<div class="mt-8 text-center text-xl">
  <span class="text-purple-500 font-bold">Quality of context determines success</span>
</div>

<!--
The shift from prompt engineering to context engineering is fundamental. It's not about crafting the perfect prompt - it's about providing the right information at the right time.
-->

---

# The 7 Fundamental Building Blocks

<div class="grid grid-cols-2 gap-4 mt-6">
  <div class="p-4 bg-blue-500 bg-opacity-10 rounded">
    <h3 class="text-blue-500">1. Intelligence 🧠</h3>
    <p class="text-sm">LLM processing and reasoning</p>
  </div>

  <div class="p-4 bg-green-500 bg-opacity-10 rounded">
    <h3 class="text-green-500">2. Memory 💾</h3>
    <p class="text-sm">Conversation state persistence</p>
  </div>

  <div class="p-4 bg-purple-500 bg-opacity-10 rounded">
    <h3 class="text-purple-500">3. Tools 🔧</h3>
    <p class="text-sm">External system integrations</p>
  </div>

  <div class="p-4 bg-yellow-500 bg-opacity-10 rounded">
    <h3 class="text-yellow-500">4. Validation ✓</h3>
    <p class="text-sm">Structured output enforcement</p>
  </div>

  <div class="p-4 bg-red-500 bg-opacity-10 rounded">
    <h3 class="text-red-500">5. Control 🚦</h3>
    <p class="text-sm">Deterministic routing logic</p>
  </div>

  <div class="p-4 bg-orange-500 bg-opacity-10 rounded">
    <h3 class="text-orange-500">6. Recovery 🔄</h3>
    <p class="text-sm">Error handling and retries</p>
  </div>

  <div class="p-4 bg-indigo-500 bg-opacity-10 rounded">
    <h3 class="text-indigo-500">7. Feedback 👤</h3>
    <p class="text-sm">Human oversight workflows</p>
  </div>
</div>

<div class="mt-6 text-center text-gray-400">
  Every agent is built from these primitives
</div>

<!--
These seven building blocks are all you need. Everything else is just combinations and variations of these fundamental components.
-->

---

# Building Block 1: Intelligence

```python
from openai import OpenAI

def basic_intelligence(prompt: str) -> str:
    """The only truly 'AI' component"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

<div class="mt-8">
  <h3 class="text-blue-500 mb-4">Key Points</h3>

- Text in → Think → Text out
- Without this, you just have regular software
- Everything else is scaffolding around this core

</div>

<!--
Intelligence is the core - it's what makes your system "AI". But notice how simple it is. The complexity comes from everything we build around it.
-->

---

# Building Block 2: Memory

```python
# Without memory - LLM forgets everything
joke = ask_llm("Tell me a joke")
followup = ask_llm("What was my previous question?")  # ❌ No idea!

# With memory - pass conversation history
messages = [
    {"role": "user", "content": "Tell me a joke"},
    {"role": "assistant", "content": joke},
    {"role": "user", "content": "What was my previous question?"}
]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages  # ✅ Full context
)
```

<div class="mt-8 text-xl text-center text-gray-400">
  LLMs are stateless - memory is just passing conversation history
</div>

<!--
LLMs don't remember anything between calls. Memory is just manually passing the conversation history each time. It's state management, like in any web app.
-->

---

# Building Block 3: Tools

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }
    }
}]

# LLM decides: "I need to call get_weather for Paris"
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)
```

<div class="mt-6 text-center text-xl">
  Move beyond text generation → Actually do things
</div>

<!--
Tools let your LLM interact with the world - call APIs, query databases, control systems. The LLM decides what to call and provides the arguments.
-->

---

# Building Block 4: Validation

```python
from pydantic import BaseModel

class TaskOutput(BaseModel):
    task: str
    priority: int
    deadline: datetime

# Force structured output
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=TaskOutput  # ✅ Guaranteed structure
)

task = response.parsed  # Type-safe, validated
```

<div class="mt-8 p-4 bg-yellow-500 bg-opacity-10 rounded">
  <h3 class="text-yellow-500">Why This Matters</h3>

- LLMs are probabilistic → outputs vary
- Downstream code needs consistent structure
- Validation + retry = reliability

</div>

<!--
Validation ensures your LLM outputs match expected schemas. This is critical for reliability - you can't build robust systems on unpredictable outputs.
-->

---

# Building Block 5: Control

```python
def route_request(user_input: str):
    """Deterministic routing - don't let LLM decide everything"""

    intent = classify_intent(user_input)  # LLM classifies

    # Regular code handles routing
    if intent == "complaint":
        return handle_complaint(user_input)
    elif intent == "technical":
        return technical_support(user_input)
    else:
        return general_response(user_input)
```

<div class="grid grid-cols-2 gap-8 mt-8">
  <div class="text-green-500">
    ✅ **Use LLMs for**: Classification, understanding intent
  </div>
  <div class="text-red-500">
    ❌ **Use code for**: Routing, business logic, control flow
  </div>
</div>

<!--
You don't want your LLM making every decision. Use it for what it's good at - understanding and classification. Use regular code for control flow.
-->

---

# Building Block 6: Recovery

```python
async def resilient_llm_call(prompt: str, max_retries: int = 3):
    """Things WILL fail - be ready"""

    for attempt in range(max_retries):
        try:
            response = await llm.complete(prompt)
            if validate_response(response):
                return response
        except RateLimitError:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except APIError as e:
            log_error(e)
            if attempt == max_retries - 1:
                return FALLBACK_RESPONSE

    return "I'm having trouble right now. Please try again."
```

<div class="mt-8 text-center">
  <div class="text-2xl text-red-500 mb-4">Things that WILL go wrong:</div>
  <div class="text-lg text-gray-400">APIs down • Rate limits • Nonsense outputs • Timeouts</div>
</div>

<!--
Production systems fail. APIs go down, rate limits hit, LLMs return garbage. Your system must handle these gracefully with retries, fallbacks, and clear error messages.
-->

---

# Building Block 7: Feedback

```python
def process_with_approval(user_request: str):
    """Some decisions need human oversight"""

    # Generate response
    draft = llm.generate_email(user_request)

    # Human review required
    print(f"Draft email:\n{draft}\n")
    approval = input("Send this email? (y/n): ")

    if approval.lower() == 'y':
        send_email(draft)
        return "Email sent successfully"
    else:
        return "Email cancelled by user"
```

<div class="mt-8 p-4 bg-purple-500 bg-opacity-10 rounded">
  <h3 class="text-purple-500 mb-2">When to Add Human Oversight</h3>

- High-stakes decisions (payments, emails to customers)
- Content generation that represents your brand
- Actions that can't be undone
- Complex judgments requiring context

</div>

<!--
Not everything should be fully automated. Some decisions are too important or complex. Build in approval workflows where humans review before execution.
-->

---

layout: center
---

# Common Agentic Patterns

<div class="text-2xl text-gray-400 mt-8">
From simple to complex
</div>

<!--
Now let's look at common patterns built from these blocks. These are proven architectures used in production systems.
-->

---

# Pattern 1: Sequential Workflows

```python
async def sequential_story_generation(topic: str):
    """Each step builds on the previous output"""

    # Step 1: Generate outline
    outline = await generate_outline(topic)

    # Step 2: Validate outline quality
    if not is_quality_outline(outline):
        outline = await improve_outline(outline)

    # Step 3: Write the story
    story = await write_story(outline)

    # Step 4: Polish final output
    final = await polish_story(story)

    return final
```

<div class="mt-8">
  <div class="text-xl mb-4">Use Cases:</div>

- Document generation (reports, articles)
- Data transformation pipelines
- Multi-stage content creation
- Step-by-step problem solving

</div>

<!--
Sequential workflows chain LLM calls where each step processes the output of the previous one. Great for tasks that naturally break down into stages.
-->

---

# Pattern 2: Routing and Handoffs

```python
# Main router agent
router_agent = Agent(
    instructions="Route to the appropriate specialist",
    handoffs=[spanish_agent, french_agent, technical_agent]
)

# Specialized agents
spanish_agent = Agent(
    instructions="You only speak Spanish",
    description="Handles Spanish language requests"
)

technical_agent = Agent(
    instructions="You are a technical expert",
    description="Handles technical questions"
)

# Usage
async def handle_request(message):
    agent = router_agent
    while True:
        result = await agent.run(message)
        if result.handoff:
            agent = result.handoff  # Specialist takes over
        else:
            return result
```

<!--
Routing patterns use a front-line agent to classify and hand off to specialists. The specialist "takes over" the conversation from that point.
-->

---

# Pattern 3: Tool Use

```python
def weather_agent():
    """Agent that can check weather and make recommendations"""

    tools = [
        get_current_weather,
        get_forecast,
        search_activities
    ]

    # User: "Should I go hiking tomorrow in Denver?"

    # Agent thinks: Need weather info first
    weather = get_forecast("Denver", "tomorrow")

    # Agent thinks: Weather looks good, find trails
    trails = search_activities("hiking trails Denver")

    # Agent responds with recommendation
    return f"Tomorrow looks great for hiking! {weather}. Try {trails[0]}"
```

<div class="mt-8 text-center">
  <div class="text-xl text-blue-500">LLM decides which tools + Tool provides real data</div>
  <div class="text-lg text-gray-400 mt-2">= Intelligent actions</div>
</div>

<!--
Tool use patterns let the LLM decide what external resources it needs. It's not just generating text - it's gathering information and taking actions.
-->

---

# Pattern 4: Reflection and Iteration

```python
async def reflective_generation(request: str, max_attempts: int = 5):
    """Generate, evaluate, and improve iteratively"""

    for attempt in range(max_attempts):
        # Generate
        output = await generator_agent.run(request)

        # Evaluate
        evaluation = await evaluator_agent.evaluate(output)

        if evaluation.score >= 0.8:
            return output

        # Provide feedback for next iteration
        request = f"""
        Previous output: {output}
        Feedback: {evaluation.feedback}
        Please improve based on feedback.
        """

    return output  # Return best effort
```

<div class="mt-6 text-center">
  <div class="text-xl">🔄 Generate → Evaluate → Improve → Repeat</div>
</div>

<!--
Reflection patterns use one LLM to evaluate another's output. This self-improvement loop can dramatically increase quality, especially for creative tasks.
-->

---

# Pattern 5: Planning and Decomposition

```python
async def research_with_planning(question: str):
    """Break complex tasks into subtasks"""

    # Step 1: Create research plan
    plan = await planner_agent.create_plan(question)
    # Output: ["Find statistics", "Check recent news", "Analyze trends"]

    # Step 2: Execute each subtask
    results = []
    for task in plan.tasks:
        result = await execute_subtask(task)
        results.append(result)

    # Step 3: Synthesize findings
    answer = await synthesizer_agent.combine(results)

    return answer
```

<div class="mt-8 p-4 bg-green-500 bg-opacity-10 rounded">
  Dynamic planning for complex problems where steps aren't known in advance
</div>

<!--
Planning patterns let the LLM break down complex tasks dynamically. Unlike sequential workflows, the steps aren't predefined - the LLM figures them out.
-->

---

# Pattern 6: Multi-Agent Collaboration

```python
# Only use when truly necessary!
orchestrator = Agent("Coordinate specialized agents")
researcher = Agent("Find and analyze information")
writer = Agent("Create polished content")
reviewer = Agent("Check quality and accuracy")

async def collaborative_report(topic: str):
    # Orchestrator manages the workflow
    research = await researcher.gather_info(topic)
    draft = await writer.create_draft(research)
    feedback = await reviewer.review(draft)
    final = await writer.revise(draft, feedback)

    return final
```

<div class="mt-8 p-4 bg-red-500 bg-opacity-10 rounded">
  ⚠️ **Warning**: Multi-agent systems are complex. Only use when a single agent truly can't handle the task.
</div>

<!--
Multi-agent patterns coordinate specialists for complex tasks. But beware - they're hard to debug and often unnecessary. Start simple.
-->

---

# Lessons from Claude Code: Simplicity Wins

<div class="grid grid-cols-2 gap-8 mt-6">
  <div>
    <h3 class="text-blue-500 mb-4">Architecture Insights</h3>

    - **Single main thread** (not multi-agent chaos)
    - **Simple tools** over complex abstractions
    - **50% of calls** use cheaper models
    - **One main loop** with sub-agents max depth 1
  </div>

  <div>
    <h3 class="text-green-500 mb-4">Tool Design</h3>

    ```python
    # High-frequency → dedicated tool
    tools = [
        Edit(),    # Used constantly
        Read(),    # Used constantly
        Search(),  # Used often
    ]

    # Low-frequency → generic bash
    bash("git commit -m 'message'")
    ```
  </div>
</div>

<div class="mt-8 text-center p-4 bg-purple-500 bg-opacity-10 rounded">
  💡 Resist the urge to over-engineer. Good harness for the model + let it cook!
</div>

<!--
Claude Code is one of the most successful AI coding assistants. Its secret? Radical simplicity. One thread, simple tools, smart use of cheaper models.
-->

---

# Building Without Frameworks

```python
# This is often all you need:
import openai
import json
from typing import Dict, Any

class SimpleAgent:
    def __init__(self, instructions: str):
        self.client = openai.OpenAI()
        self.instructions = instructions
        self.messages = []

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.instructions},
                *self.messages
            ]
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message
```

<div class="mt-6 text-center text-xl text-gray-400">
  Direct API calls = Full control, easy debugging, no magic
</div>

<!--
You don't need LangChain or other frameworks to start. Direct API calls give you full control and understanding of what's happening.
-->

---

# The Framework Tradeoff for Production

<div class="grid grid-cols-2 gap-8 mt-6">
  <div>
    <h3 class="text-red-500 mb-4">Start Raw (Learning/Prototyping)</h3>

    ```python
    # Direct API calls
    response = client.chat.completions.create(...)

    # Manual logging
    print(f"Called LLM: {response}")

    # Simple error handling
    try:
        ...
    except Exception as e:
        print(f"Error: {e}")
    ```

    ✅ Full understanding<br/>
    ✅ Easy debugging<br/>
    ❌ Manual everything
  </div>

  <div>
    <h3 class="text-green-500 mb-4">Add Framework (Production/Scale)</h3>

    ```python
    # Automatic observability
    @trace
    def agent_step():
        # Framework handles:
        # - Distributed tracing
        # - Metrics collection
        # - Log aggregation
        # - Error reporting
        pass
    ```

    ✅ Built-in observability<br/>
    ✅ Production infrastructure<br/>
    ❌ Abstraction overhead
  </div>
</div>

<div class="mt-8 text-center p-4 bg-blue-500 bg-opacity-10 rounded">
  **Key Insight**: Frameworks shine for infrastructure (logging, tracing, monitoring), not agent logic
</div>

<!--
The real value of frameworks isn't in agent logic - it's in production infrastructure. Start raw to understand, adopt frameworks when you need observability at scale.
-->

---

# Error Handling and Recovery

```python
class RobustAgent:
    async def call_with_fallback(self, prompt: str):
        strategies = [
            (self.primary_llm, "gpt-4o"),
            (self.backup_llm, "gpt-3.5-turbo"),
            (self.local_llm, "llama-3"),
            (self.cached_response, None),
            (self.default_response, None)
        ]

        for strategy, model in strategies:
            try:
                return await strategy(prompt, model)
            except Exception as e:
                log_error(f"Failed with {model}: {e}")
                continue

        return "I'm unable to help right now. Please try again later."
```

<div class="mt-8 text-2xl text-center">
  <span class="text-red-500">Expect failure</span> →
  <span class="text-yellow-500">Plan for it</span> →
  <span class="text-green-500">Degrade gracefully</span>
</div>

<!--
Production systems must handle failures gracefully. Have backup models, cached responses, and clear error messages. Users prefer a degraded experience over no experience.
-->

---

# Validation and Guardrails

```python
# Input guardrails
@input_guardrail
async def check_harmful_content(input_text: str):
    if contains_pii(input_text):
        raise ValidationError("Please remove personal information")

    if is_prompt_injection(input_text):
        raise SecurityError("Invalid input detected")

# Output guardrails
@output_guardrail
async def validate_response(output: str):
    if contains_sensitive_data(output):
        return "[REDACTED]"

    if not appropriate_for_audience(output):
        return regenerate_with_constraints(output)

    return output

# Usage
@with_guardrails(input=[check_harmful_content],
                 output=[validate_response])
async def safe_agent_call(user_input: str):
    return await agent.process(user_input)
```

<!--
Guardrails protect your system and users. Input guardrails catch harmful requests. Output guardrails ensure responses are safe and appropriate.
-->

---

# When NOT to Use Agents

<div class="grid grid-cols-2 gap-8 mt-8">
  <div class="p-6 bg-red-500 bg-opacity-10 rounded">
    <h3 class="text-red-500 mb-4">❌ Bad Fit for Agents</h3>

    - Known steps, deterministic process
    - Simple Q&A from documentation
    - Mathematical calculations
    - Data transformations with fixed rules
    - Real-time/low-latency requirements

    <div class="mt-4 text-sm text-gray-400">
      Use regular code or simple LLM calls
    </div>
  </div>

  <div class="p-6 bg-green-500 bg-opacity-10 rounded">
    <h3 class="text-green-500 mb-4">✅ Good Fit for Agents</h3>

    - Complex reasoning required
    - Multiple tools needed dynamically
    - Iterative refinement beneficial
    - Planning required for unknown steps
    - Human-like judgment needed

    <div class="mt-4 text-sm text-gray-400">
      Agents add value here
    </div>
  </div>
</div>

<div class="mt-8 text-center text-xl">
  <span class="text-purple-500">Agents trade latency + cost for capability</span>
</div>

<!--
Not everything needs an agent. They're expensive and slow. Use them when the complexity justifies the cost, not for simple deterministic tasks.
-->

---

layout: center
---

# Start Simple, Iterate Deliberately

<div class="text-6xl mt-8">
  1️⃣ → 🔧 → ♾️
</div>

<div class="grid grid-cols-3 gap-4 mt-8 text-lg">
  <div class="text-center">
    <div class="text-blue-500 font-bold">Single LLM call</div>
    <div class="text-sm text-gray-400">Start here always</div>
  </div>

  <div class="text-center">
    <div class="text-green-500 font-bold">Add tools gradually</div>
    <div class="text-sm text-gray-400">When needed</div>
  </div>

  <div class="text-center">
    <div class="text-purple-500 font-bold">Multi-agent last</div>
    <div class="text-sm text-gray-400">When proven necessary</div>
  </div>
</div>

<div class="mt-12 text-xl text-gray-400 text-center">
  Complexity should be earned, not assumed
</div>

<!--
This is the most important principle. Start with the simplest solution. Add complexity only when you've proven simpler approaches don't work.
-->

---

layout: section
---

# Lab Exercises

<div class="text-2xl text-gray-400 mt-8">
  Let's build some agents!
</div>

<!--
Now we'll put these concepts into practice with hands-on exercises building real agents using the patterns we've discussed.
-->

---

# Lab 1: Build a Simple Agent

```python
# No frameworks - just Python and API calls
import openai
from typing import List, Dict

class ConversationAgent:
    """A minimal agent with memory"""

    def __init__(self, system_prompt: str):
        self.client = openai.OpenAI()
        self.system_prompt = system_prompt
        self.conversation: List[Dict[str, str]] = []

    def chat(self, user_input: str) -> str:
        # Add user message to history
        self.conversation.append({"role": "user", "content": user_input})

        # Call LLM with full context
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Start with cheaper model
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.conversation
            ]
        )

        # Extract and store response
        assistant_reply = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

# Usage
agent = ConversationAgent("You are a helpful data science tutor")
print(agent.chat("Explain gradient descent"))
print(agent.chat("Can you give an example?"))  # Remembers context!
```

<!--
Start with the absolute basics. This simple agent has memory and can maintain context across conversations. No frameworks needed.
-->

---

# Lab 1: Add Sequential Workflow

```python
class StoryAgent(ConversationAgent):
    """Extend our agent with sequential workflow pattern"""

    def generate_story(self, topic: str) -> Dict[str, str]:
        results = {}

        # Step 1: Generate outline
        outline_prompt = f"Create a short story outline about: {topic}"
        results['outline'] = self.chat(outline_prompt)

        # Step 2: Write first draft
        draft_prompt = f"Write a story based on this outline: {results['outline']}"
        results['draft'] = self.chat(draft_prompt)

        # Step 3: Add details
        enhance_prompt = f"Enhance this story with vivid details: {results['draft']}"
        results['final'] = self.chat(enhance_prompt)

        return results

# Test the workflow
story_agent = StoryAgent("You are a creative writer")
story = story_agent.generate_story("a robot learning to paint")

print("Outline:", story['outline'][:100], "...")
print("Draft:", story['draft'][:100], "...")
print("Final:", story['final'][:200], "...")
```

<!--
Now we add a sequential workflow. Each step builds on the previous one. This is how you'd build document generation or multi-stage processing.
-->

---

# Lab 1: Implement Tool Use

```python
import json
import requests

class WeatherAgent(ConversationAgent):
    """Add tool use capability"""

    def get_weather(self, city: str) -> str:
        """Actual tool that fetches weather"""
        # Using a free weather API
        response = requests.get(f"http://wttr.in/{city}?format=j1")
        data = response.json()
        return f"Current temp: {data['current_condition'][0]['temp_C']}°C"

    def chat_with_tools(self, user_input: str) -> str:
        # Check if weather info is needed
        check_prompt = f"""
        Does this request need weather information? Reply with JSON:
        {{"needs_weather": true/false, "city": "city name or null"}}

        User request: {user_input}
        """

        check_response = self.chat(check_prompt)

        try:
            decision = json.loads(check_response)
            if decision.get('needs_weather') and decision.get('city'):
                weather = self.get_weather(decision['city'])
                return self.chat(f"User asked: {user_input}\nWeather: {weather}")
        except:
            pass

        return self.chat(user_input)

# Test it
weather_agent = WeatherAgent("You are a helpful weather assistant")
print(weather_agent.chat_with_tools("What's the weather in London?"))
print(weather_agent.chat_with_tools("How does rain form?"))  # No tool needed
```

<!--
Tool use lets your agent interact with external systems. Here we're checking weather, but the pattern works for any API or service.
-->

---

# Lab 2: Add Reflection Pattern

```python
class ReflectiveAgent(ConversationAgent):
    """Agent that can evaluate and improve its outputs"""

    def generate_with_reflection(self, task: str, max_iterations: int = 3):
        current_output = None

        for i in range(max_iterations):
            if current_output is None:
                # First attempt
                current_output = self.chat(f"Task: {task}")
            else:
                # Improve based on self-evaluation
                improve_prompt = f"""
                Previous output: {current_output}
                Evaluation: {evaluation}

                Please improve your response based on the evaluation.
                """
                current_output = self.chat(improve_prompt)

            # Self-evaluate
            eval_prompt = f"""
            Evaluate this output for the task '{task}':
            {current_output}

            Rate 1-10 and explain what could be better.
            Format: {{"score": N, "feedback": "..."}}
            """

            evaluation = self.chat(eval_prompt)

            try:
                eval_data = json.loads(evaluation)
                if eval_data.get('score', 0) >= 8:
                    print(f"✅ Accepted after {i+1} iterations")
                    break
            except:
                pass

        return current_output

# Test reflection
reflective = ReflectiveAgent("You are a code reviewer")
code = reflective.generate_with_reflection(
    "Write a Python function to check if a string is a palindrome"
)
print(code)
```

<!--
Reflection dramatically improves quality. The agent evaluates its own work and iterates until it meets quality standards.
-->

---

# Lab 3: Implement Error Recovery

```python
import time
from typing import Optional

class ResilientAgent(ConversationAgent):
    """Agent with comprehensive error handling"""

    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)
        self.fallback_responses = {
            'general': "I'm having trouble processing that request.",
            'rate_limit': "I'm getting too many requests. Please wait a moment.",
            'timeout': "This is taking longer than expected.",
        }

    def chat_with_retry(
        self,
        user_input: str,
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> str:

        last_error = None

        for attempt in range(max_retries):
            try:
                # Simulate timeout handling
                start_time = time.time()

                # Try primary model
                response = self._try_model(user_input, "gpt-4o", timeout)
                if response:
                    return response

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")

                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # All retries failed - use fallback
        return self._get_fallback(last_error)

    def _try_model(self, input: str, model: str, timeout: float) -> Optional[str]:
        """Try a specific model with timeout"""
        # In production, use asyncio with actual timeout
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input}
                ],
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise Exception("rate_limit")
            raise

    def _get_fallback(self, error: Exception) -> str:
        """Return appropriate fallback response"""
        error_str = str(error).lower()

        if "rate_limit" in error_str:
            return self.fallback_responses['rate_limit']
        elif "timeout" in error_str:
            return self.fallback_responses['timeout']
        else:
            return self.fallback_responses['general']

# Test resilience
resilient = ResilientAgent("You are a helpful assistant")
response = resilient.chat_with_retry("Tell me about machine learning")
print(response)
```

<!--
Production agents must handle failures gracefully. This implementation shows retry logic, exponential backoff, and fallback responses.
-->

---

# Lab 3: Add Human Feedback

```python
class SupervisedAgent(ConversationAgent):
    """Agent with human-in-the-loop feedback"""

    def __init__(self, system_prompt: str, require_approval: bool = False):
        super().__init__(system_prompt)
        self.require_approval = require_approval
        self.feedback_history = []

    def chat_with_feedback(self, user_input: str) -> str:
        # Generate initial response
        response = self.chat(user_input)

        # Check if approval required
        if self.require_approval:
            response = self._get_approval(response, user_input)

        return response

    def _get_approval(self, response: str, original_input: str) -> str:
        """Get human approval before finalizing"""
        print("\n" + "="*50)
        print("DRAFT RESPONSE:")
        print(response)
        print("="*50)

        while True:
            feedback = input("\n[A]pprove, [R]evise, or [C]ancel? ").lower()

            if feedback == 'a':
                print("✅ Response approved")
                return response

            elif feedback == 'r':
                revision_request = input("What should be changed? ")
                self.feedback_history.append({
                    'original': response,
                    'feedback': revision_request
                })

                # Generate revised response
                revision_prompt = f"""
                Original request: {original_input}
                Draft response: {response}
                Feedback: {revision_request}

                Please revise the response based on the feedback.
                """
                response = self.chat(revision_prompt)
                print("\n🔄 Revised response generated")

            elif feedback == 'c':
                return "Request cancelled by user."

    def show_feedback_stats(self):
        """Display feedback statistics"""
        if not self.feedback_history:
            print("No feedback collected yet")
            return

        print(f"\nFeedback Statistics:")
        print(f"Total revisions: {len(self.feedback_history)}")

        # Analyze common feedback themes
        common_feedback = {}
        for item in self.feedback_history:
            words = item['feedback'].lower().split()
            for word in words:
                common_feedback[word] = common_feedback.get(word, 0) + 1

        print("Common feedback terms:", dict(sorted(
            common_feedback.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]))

# Test supervised agent
supervised = SupervisedAgent(
    "You are writing professional emails",
    require_approval=True
)

# Simulate email generation
email = supervised.chat_with_feedback(
    "Write an email declining a meeting invitation politely"
)
print("\nFinal email:", email)

supervised.show_feedback_stats()
```

<!--
Human feedback is crucial for high-stakes outputs. This pattern allows review and revision before any action is taken.
-->

---

layout: center
---

# Key Takeaways

<div class="grid grid-cols-2 gap-8 mt-8">
  <div class="space-y-4">
    <div class="flex items-start gap-3">
      <span class="text-2xl">1️⃣</span>
      <div>
        <div class="font-bold">Start simple</div>
        <div class="text-sm text-gray-400">Single LLM call → Add complexity only when proven necessary</div>
      </div>
    </div>

    <div class="flex items-start gap-3">
      <span class="text-2xl">2️⃣</span>
      <div>
        <div class="font-bold">Context > Prompts</div>
        <div class="text-sm text-gray-400">Quality of context determines success</div>
      </div>
    </div>

    <div class="flex items-start gap-3">
      <span class="text-2xl">3️⃣</span>
      <div>
        <div class="font-bold">Build blocks, not monoliths</div>
        <div class="text-sm text-gray-400">7 primitives combine into any pattern</div>
      </div>
    </div>
  </div>

  <div class="space-y-4">
    <div class="flex items-start gap-3">
      <span class="text-2xl">4️⃣</span>
      <div>
        <div class="font-bold">Expect failure</div>
        <div class="text-sm text-gray-400">Robust error handling is not optional</div>
      </div>
    </div>

    <div class="flex items-start gap-3">
      <span class="text-2xl">5️⃣</span>
      <div>
        <div class="font-bold">Use frameworks wisely</div>
        <div class="text-sm text-gray-400">For observability, not agent logic</div>
      </div>
    </div>

    <div class="flex items-start gap-3">
      <span class="text-2xl">6️⃣</span>
      <div>
        <div class="font-bold">Simplicity wins</div>
        <div class="text-sm text-gray-400">Claude Code proves this daily</div>
      </div>
    </div>
  </div>
</div>

<!--
These are the key principles to remember. Start simple, focus on context, build with composable blocks, handle failures, and resist unnecessary complexity.
-->

---

layout: section
---

# Resources

<div class="mt-8 space-y-3 text-lg">

📚 **Essential Reading**

- [Building Effective Agents](https://anthropic.com/engineering/building-effective-agents) - Anthropic
- [Multi-Agent Research System](https://anthropic.com/engineering/multi-agent-research-system) - Anthropic
- [Claude Code Analysis](https://minusx.ai/blog/decoding-claude-code/) - MinusX
- [Agentic Patterns](https://www.philschmid.de/agentic-pattern) - Phil Schmid

🔧 **Code Examples**

- [Building Blocks Repository](https://github.com/daveebbelaar/ai-cookbook/tree/main/agents/building-blocks)
- [OpenAI Agent Patterns](https://github.com/openai/openai-agents-python)
- [Agent Recipes](https://www.agentrecipes.com/)

💬 **Community**

- [Claude Developers Discord](https://discord.gg/claude)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

</div>

<!--
Here are the key resources for diving deeper. Start with Anthropic's articles - they're based on real production experience.
-->

---

layout: center
class: text-center
---

# Questions?

<div class="mt-8 text-xl text-gray-400">
Let's discuss what you're building
</div>

<div class="mt-12 text-sm text-gray-500">
Contact: [your contact info]<br/>
Slides: [repository link]
</div>

<!--
Thank you for your attention. I'm excited to hear about what you're planning to build with these concepts.
-->
