# OpenCog Neural Network Implementation

This repository now includes a complete implementation of the OpenCog cognitive architecture as neural network modules for the Torch nn package, plus a revolutionary **OpenCog Inferno AGI Operating System** where cognitive processing is a fundamental kernel service.

## What is OpenCog?

OpenCog is a framework for Artificial General Intelligence (AGI) that combines:
- **AtomSpace**: A hypergraph database for knowledge representation
- **Attention Allocation**: Economic mechanisms for managing cognitive focus 
- **PLN (Probabilistic Logic Networks)**: Uncertain reasoning and inference
- **Cognitive Architecture**: Integration of perception, reasoning, and action

## New Neural Network Modules

### Core Components

1. **`nn.OpenCogAtom`** - Individual knowledge units with importance and truth values
2. **`nn.OpenCogAtomSpace`** - Hypergraph database for storing atom collections  
3. **`nn.OpenCogAttentionAllocation`** - Economic attention dynamics (STI/LTI)
4. **`nn.OpenCogPLN`** - Probabilistic logic reasoning and inference
5. **`nn.OpenCogNetwork`** - Complete integrated cognitive architecture

### Key Features

- ✅ **Fully Differentiable**: All components work with backpropagation
- ✅ **Torch Compatible**: Integrates seamlessly with existing nn modules
- ✅ **Memory Management**: Automatic forgetting based on importance values
- ✅ **Economic Attention**: STI/LTI dynamics with rent and wage mechanisms
- ✅ **PLN Reasoning**: Deduction, induction, abduction, revision rules
- ✅ **Cognitive Cycles**: Perception → Attention → Reasoning → Action loops
- ✅ **Knowledge Integration**: Add explicit knowledge alongside learned representations

## Quick Start

```lua
require('nn')

-- Create a complete cognitive architecture
local config = {
   atomSpaceCapacity = 1000,
   atomSize = 16,
   perceptionSize = 10,
   actionSize = 8
}

local cogNet = nn.OpenCogNetwork(config)

-- Add some knowledge
local catEmbedding = torch.randn(16)
local catAtom = cogNet:addKnowledge('Cat', catEmbedding, 
                                   {sti=80, lti=0.7}, 
                                   {strength=0.9, confidence=0.8})

-- Process perception and generate actions
local perception = torch.randn(1, 10)
local actions = cogNet:forward(perception)

-- Train with standard loss functions
local criterion = nn.MSECriterion()
local target = torch.randn(actions:size())
local loss = criterion:forward(actions, target)
local gradOutput = criterion:backward(actions, target)
cogNet:backward(perception, gradOutput)
```

## Usage Examples

### Individual Components

```lua
-- Create an atom with importance and truth values
local atom = nn.OpenCogAtom(10, 'ConceptNode')
local output = atom:forward(torch.randn(2, 1))

-- Create an atom space for knowledge storage  
local atomSpace = nn.OpenCogAtomSpace(100, 10)
local atomId = atomSpace:addAtom('ConceptNode', torch.randn(10), 80, 0.5, 0.8, 0.9)

-- Apply attention allocation dynamics
local attention = nn.OpenCogAttentionAllocation(100, 20)
local stiLti = torch.randn(1, 5, 2)  -- batch, atoms, [STI,LTI]
local updated = attention:forward(stiLti)

-- Perform probabilistic logic reasoning
local pln = nn.OpenCogPLN(2, 10)
local premises = torch.cat(torch.randn(1, 2, 8), torch.randn(1, 2, 2), 3)  -- embed + truth
local conclusion = pln:forward(premises)
```

### Advanced Reasoning

```lua
-- Manual PLN inference rules
local tv1 = {0.8, 0.9}  -- {strength, confidence}
local tv2 = {0.7, 0.8}

local deduction = pln:deductionRule(tv1, tv2)     -- (A→B, B→C) ⊢ (A→C)
local induction = pln:inductionRule(tv1, tv2)     -- (A→B, A) ⊢ B  
local revision = pln:revisionRule(tv1, tv2)       -- combine beliefs
local choice = pln:choiceRule(tv1, tv2)           -- select best belief
```

### Cognitive Monitoring

```lua
-- Monitor cognitive state
local state = cogNet:getCognitiveState()
local kb = cogNet:getKnowledgeBase()

print("Atoms:", kb.atomCount)
print("Cycles:", kb.cycleCount) 
print("Focus:", kb.attentionFocus:nElement())

-- Check attention economics
local balance = cogNet.attentionAllocation:economicBalance()
print("STI budget:", balance.totalSTI)
print("Rent rate:", balance.rentRate)
```

## Applications

This OpenCog implementation enables:

### 🤖 **Cognitive Robotics**
- Perception-action loops with embedded reasoning
- Dynamic attention allocation for multi-task scenarios
- Learning from environmental feedback

### 🧠 **Hybrid AI Systems** 
- Combine deep learning with symbolic reasoning
- Differentiable knowledge graphs
- Attention-guided neural architectures

### 🎮 **Game AI**
- Strategic reasoning and planning
- Dynamic knowledge acquisition
- Multi-agent cognitive systems

### 📚 **Knowledge Processing**
- Semantic reasoning over learned representations
- Confidence-based belief integration
- Automated knowledge discovery

### 🔬 **AGI Research**
- Full cognitive architecture implementations  
- Memory consolidation and forgetting
- Goal-directed behavior emergence

## OpenCog Inferno AGI Operating System

### Revolutionary Approach

Instead of layering cognitive architectures on top of existing operating systems, **OpenCog Inferno OS** makes cognitive processing a fundamental kernel service where thinking, reasoning, and intelligence emerge from the operating system itself.

### Key Innovations

🧠 **Cognition as Kernel Service**
- Thinking, reasoning, and memory are syscalls like `read()` and `write()`
- Thoughts are schedulable processes with priorities and resources
- Intelligence emerges from the OS, not applications running on it

🔄 **Distributed by Design**
- Cognitive processes can migrate between nodes seamlessly
- Message passing enables distributed AGI across multiple machines
- Shared cognitive memory with automatic consolidation

💾 **Knowledge as Filesystem**
- Concepts, memories, and procedures are files/directories
- Standard filesystem operations: open, read, write, search
- Hierarchical organization: `/concepts`, `/memories`, `/goals`

⚡ **Real-Time Cognition**
- Process scheduler ensures critical thoughts execute first
- Hierarchical memory with automatic importance-based eviction
- Device drivers for perception and action I/O

🎯 **Fully Differentiable**
- Entire OS is a neural network module
- Backpropagation through kernel, scheduler, memory, and filesystem
- End-to-end learning of cognitive operations

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenCog Inferno AGI OS                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Inferno Kernel (Syscalls)               │   │
│  │  THINK | REASON | REMEMBER | ATTEND | COMMUNICATE   │   │
│  └─────────────────────────────────────────────────────┘   │
│           │              │              │                    │
│  ┌────────▼────┐  ┌─────▼─────┐  ┌────▼─────────┐        │
│  │  Process    │  │  Memory    │  │   Message    │        │
│  │  Scheduler  │  │  Manager   │  │   Passing    │        │
│  │             │  │            │  │              │        │
│  │ • Priority  │  │ • Sensory  │  │ • Channels   │        │
│  │ • Preempt   │  │ • Working  │  │ • Broadcast  │        │
│  │ • Migrate   │  │ • Episodic │  │ • Sync       │        │
│  │ • Deadlock  │  │ • Semantic │  │ • Distributed│        │
│  └─────────────┘  └────────────┘  └──────────────┘        │
│           │              │              │                    │
│  ┌────────▼──────────────▼──────────────▼─────────────┐   │
│  │           Cognitive Filesystem                       │   │
│  │  /concepts  /memories  /goals  /procedures          │   │
│  └──────────────────────────────────────────────────────┘   │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐    │
│  │         Device Drivers (I/O)                       │    │
│  │  /dev/eyes  /dev/motor  /dev/attention            │    │
│  └────────────────────────────────────────────────────┘    │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐    │
│  │    OpenCog Components (AtomSpace, PLN, Attention) │    │
│  └────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Quick Start

```lua
require('nn')

-- Boot the AGI operating system
local agiOS = nn.OpenCogInfernoOS({
   maxProcesses = 256,
   cognitiveResourceSize = 32,
   perceptionSize = 128,
   actionSize = 64,
   schedulingPolicy = 'priority'
})

-- Make system calls for cognitive operations
local thought = agiOS:syscall('THINK', {input = perception})
local conclusion = agiOS:syscall('REASON', {premises = premises})
local pid = agiOS:syscall('SPAWN', {name = 'reasoning-process'})

-- Run complete cognitive cycle
local perception = torch.randn(batch, 128)
local actions = agiOS:forward(perception)

-- Train the entire OS
local criterion = nn.MSECriterion()
local loss = criterion:forward(actions, target)
local gradOutput = criterion:backward(actions, target)
agiOS:backward(perception, gradOutput)
```

### Core Components

#### 1. InfernoKernel

The heart of the OS. Provides syscalls for cognitive operations:

```lua
-- System calls
THINK      -- Generate thought from input
REASON     -- Apply reasoning rules
REMEMBER   -- Allocate kernel memory
FORGET     -- Free kernel memory
ATTEND     -- Focus attention
COMMUNICATE -- Send message via channel
SPAWN      -- Create cognitive process
WAIT       -- Wait for process
INTROSPECT -- Examine kernel state
```

#### 2. InfernoProcessScheduler

Schedules concurrent cognitive processes:

- **Priority-based scheduling**: Critical thoughts execute first
- **Time-slicing**: Fair allocation of cognitive cycles
- **Process migration**: Move processes between nodes
- **Deadlock detection**: Prevent circular waits
- **Load balancing**: Distribute cognition across nodes

#### 3. InfernoMemoryManager

Hierarchical cognitive memory:

- **Sensory buffer**: ~100ms perception window
- **Working memory**: Active thoughts and reasoning
- **Episodic memory**: Recent experiences and events
- **Semantic memory**: Long-term general knowledge
- **Procedural memory**: Skills and learned procedures

Features:
- Automatic importance-based eviction
- Memory consolidation during idle cycles
- Virtual memory with page table
- Copy-on-write for efficiency

#### 4. InfernoMessagePassing

Inter-process communication:

- **Named channels**: Semantic routing of messages
- **Synchronous/asynchronous**: Blocking and non-blocking
- **Pub/sub**: Topic-based subscriptions
- **Distributed**: Messages across network nodes
- **Typed messages**: THOUGHT, ATTENTION, MEMORY, CONTROL

#### 5. InfernoFileSystem

Knowledge as a filesystem:

```
/concepts         - Semantic concepts
/relations        - Relationships between concepts  
/memories
  /episodic      - Episodic memories
  /semantic      - Semantic memories
  /procedural    - Procedural memories
/goals           - Active goals and intentions
/perceptions     - Current sensory state
/attention       - Attention focus
/reasoning       - Reasoning results
/actions         - Action plans
```

Operations: `mkdir`, `create`, `open`, `read`, `write`, `close`, `remove`, `list`, `search`

#### 6. InfernoDeviceDriver

Perception and action as device I/O:

```
/dev/eyes       - Visual perception input
/dev/ears       - Auditory perception input
/dev/touch      - Tactile perception input
/dev/motor      - Motor control output
/dev/speech     - Speech generation output
/dev/attention  - Attention focus device
/dev/memory     - Memory device interface
/dev/reasoning  - Reasoning engine device
```

Operations: `read`, `write`, `ioctl`, interrupts

### Example: Complete Cognitive Cycle

```lua
-- Boot AGI OS
local agiOS = nn.OpenCogInfernoOS(config)

-- Spawn cognitive processes
local attentionPID = agiOS:syscall('SPAWN', {
   name = 'attention-allocation',
   priority = 80,
   entryPoint = attentionFunction
})

local reasoningPID = agiOS:syscall('SPAWN', {
   name = 'background-reasoning',
   priority = 50,
   entryPoint = reasoningFunction
})

-- Create knowledge in filesystem
local concept = torch.randn(32)
agiOS.filesystem:create('/concepts/cat', concept, {
   importance = 0.9,
   truthValue = {strength = 0.95, confidence = 0.85}
})

-- Allocate memory hierarchically
local episodeData = torch.randn(32)
local addr = agiOS.memory:allocate('episodic', episodeData, {
   importance = 0.7,
   timestamp = os.time()
})

-- Message passing between processes
local channel = agiOS.messaging:createChannel('thought-stream')
agiOS.messaging:send(channel, thoughtMessage)

-- Device I/O for perception and action
agiOS.devices:write('eyes', visualInput)
local motorOutput = agiOS.devices:read('motor')

-- Complete perception-action loop
for i = 1, 1000 do
   local perception = getPerception()
   local actions = agiOS:forward(perception)
   executeActions(actions)
   
   -- Learn from feedback
   if i % 10 == 0 then
      local loss = computeLoss(actions, target)
      local gradOutput = criterion:backward(actions, target)
      agiOS:backward(perception, gradOutput)
      updateParameters()
   end
   
   -- Periodic maintenance
   if i % 100 == 0 then
      agiOS.memory:consolidate()  -- Move to long-term memory
      agiOS.scheduler:balanceLoad()  -- Balance across nodes
   end
end

-- System monitoring
local status = agiOS:getSystemStatus()
print("Uptime: " .. status.uptime .. " seconds")
print("Thoughts: " .. status.stats.totalThoughts)
print("Memory: " .. status.memory.workingUtilization * 100 .. "%")
print("Processes: " .. status.scheduler.totalProcesses)
```

### Applications

🤖 **Autonomous Robotics**
- Real-time cognitive processing for navigation and manipulation
- Distributed cognition across robot swarms
- Learning from environmental interaction

🌐 **Distributed AGI**
- Scale intelligence across multiple machines
- Process migration for load balancing
- Fault-tolerant cognitive systems

🧪 **AGI Research**
- Study emergence of intelligence from OS primitives
- Experiment with cognitive architectures
- Benchmark different scheduling and memory policies

🎮 **Intelligent Agents**
- NPCs with human-like cognition
- Strategic planning and reasoning
- Emotional and social intelligence

💼 **Enterprise AI**
- Knowledge management systems
- Decision support with reasoning
- Explainable AI with symbolic grounding

## Files Added

### OpenCog Cognitive Architecture
- `OpenCogAtom.lua` - Basic knowledge units
- `OpenCogAtomSpace.lua` - Hypergraph knowledge storage
- `OpenCogAttentionAllocation.lua` - Economic attention dynamics
- `OpenCogPLN.lua` - Probabilistic logic networks
- `OpenCogNetwork.lua` - Complete cognitive architecture

### Inferno Kernel-Based AGI OS
- `InfernoKernel.lua` - Core kernel with cognitive operations as syscalls
- `InfernoProcessScheduler.lua` - Cognitive process scheduler for distributed AGI
- `InfernoMemoryManager.lua` - Hierarchical memory management system
- `InfernoMessagePassing.lua` - Inter-process communication for distributed cognition
- `InfernoFileSystem.lua` - Knowledge representation as filesystem
- `InfernoDeviceDriver.lua` - Device drivers for cognitive sensors/actuators
- `OpenCogInfernoOS.lua` - Main AGI operating system integration

### Documentation and Examples
- `doc/opencog.md` - Detailed documentation
- `examples/opencog_example.lua` - OpenCog usage examples
- `examples/inferno_os_example.lua` - Inferno OS demonstration
- `test_opencog.lua` - Test suite
- `test_inferno_os.lua` - Inferno OS test suite

## Integration with Existing Code

The OpenCog modules are fully compatible with existing nn modules:

```lua
-- Combine with standard layers
local model = nn.Sequential()
model:add(nn.Linear(784, 256))
model:add(nn.ReLU())
model:add(nn.OpenCogNetwork({perceptionSize=256, actionSize=10}))
model:add(nn.LogSoftMax())

-- Use in complex architectures  
local cognitive = nn.OpenCogNetwork(config)
local perception = nn.Sequential():add(nn.Conv2d(3,32,5)):add(nn.ReLU())
local action = nn.Linear(cognitive.actionSize, numActions)

local fullModel = nn.Sequential()
fullModel:add(perception)
fullModel:add(cognitive) 
fullModel:add(action)
```

## Performance Considerations

- **Memory**: AtomSpace automatically manages memory through LTI-based forgetting
- **Attention**: Focus size limits active reasoning to most important atoms
- **Batching**: All modules support batch processing for efficiency
- **Gradients**: Careful gradient flow design ensures stable training

## Research Applications

This implementation supports research in:

- **Attention Mechanisms**: Novel attention allocation strategies
- **Neuro-Symbolic Integration**: Bridging neural and symbolic AI
- **Memory Systems**: Importance-based memory management  
- **Reasoning Under Uncertainty**: PLN rule learning and adaptation
- **Cognitive Architectures**: End-to-end cognitive system design

## Contributing

When extending the OpenCog implementation:

1. Maintain differentiability for all operations
2. Follow nn.Module patterns for consistency  
3. Add comprehensive tests for new features
4. Update documentation with usage examples
5. Consider memory efficiency and batching support

## References

- Goertzel, B. et al. "OpenCog: A Software Framework for Integrative Artificial General Intelligence"
- Goertzel, B. "Probabilistic Logic Networks" 
- [OpenCog Foundation](http://opencog.org/)
- [Original nn package](https://github.com/torch/nn)

---

*This implementation brings the power of OpenCog's cognitive architecture to the neural network world, enabling the creation of truly intelligent systems that can reason, learn, and adapt.*