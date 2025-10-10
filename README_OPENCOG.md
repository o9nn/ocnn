# OpenCog Neural Network Implementation

This repository now includes a complete implementation of the OpenCog cognitive architecture as neural network modules for the Torch nn package.

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
6. **`nn.OpenCogMetrics`** - Real-time cognitive performance monitoring
7. **`nn.OpenCogWorkingMemory`** - Goal management and episodic memory

### Key Features

- ✅ **Fully Differentiable**: All components work with backpropagation
- ✅ **Torch Compatible**: Integrates seamlessly with existing nn modules
- ✅ **Memory Management**: Automatic forgetting based on importance values
- ✅ **Economic Attention**: STI/LTI dynamics with rent and wage mechanisms
- ✅ **PLN Reasoning**: Deduction, induction, abduction, revision rules
- ✅ **Cognitive Cycles**: Perception → Attention → Reasoning → Action loops
- ✅ **Knowledge Integration**: Add explicit knowledge alongside learned representations
- ✅ **Advanced PLN Rules**: Similarity, contraposition, hypothetical, intensional reasoning
- ✅ **Working Memory**: Goal stack, episodic buffer, context tracking, predictions
- ✅ **Cognitive Metrics**: Real-time monitoring of attention, memory, reasoning, learning
- ✅ **Goal-Directed Behavior**: Hierarchical goal management and achievement tracking
- ✅ **Episodic Learning**: Experience replay and memory consolidation
- ✅ **Predictive Capabilities**: Future state prediction based on past experiences

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

## Architecture Overview

```
Perception → AtomSpace → Attention → PLN → Action
     ↑           ↓          ↓        ↓       ↓
   Input    Knowledge    Focus   Reasoning  Output
            Storage    Management           
                         ↓
                   Memory & Learning
```

## Files Added

- `OpenCogAtom.lua` - Basic knowledge units
- `OpenCogAtomSpace.lua` - Hypergraph knowledge storage
- `OpenCogAttentionAllocation.lua` - Economic attention dynamics
- `OpenCogPLN.lua` - Probabilistic logic networks
- `OpenCogNetwork.lua` - Complete cognitive architecture
- `doc/opencog.md` - Detailed documentation
- `examples/opencog_example.lua` - Usage examples
- `test_opencog.lua` - Test suite

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

## Recent Enhancements (v2.0)

### 🧠 Advanced Reasoning
- **Extended PLN Rules**: Similarity, contraposition, hypothetical, and intensional reasoning
- **Rule Chaining**: Complex inference sequences with multiple rule types
- **Cached Inference**: Performance optimization with result caching

### 💭 Working Memory System
- **Goal Stack**: Hierarchical goal management with priorities and achievement tracking
- **Episodic Buffer**: Experience replay with context and reward information
- **Predictive System**: Future state prediction based on past experiences
- **Context Tracking**: Dynamic context updates with history maintenance

### 📊 Cognitive Metrics
- **Real-time Monitoring**: Attention efficiency, memory utilization, reasoning success
- **Performance Analytics**: Learning rates, adaptation speed, conceptual coherence
- **System Health**: Gradient health, parameter stability, economic balance
- **Detailed Reporting**: Comprehensive cognitive state analysis

### 🛡️ Error Handling & Validation
- **Input Validation**: Truth values, embeddings, batch dimensions, inference rules
- **Health Checks**: System diagnostics and problem detection
- **Gradient Monitoring**: NaN/infinity detection, exploding gradient alerts
- **Memory Safety**: Capacity validation and boundary checking

### ⚡ Performance Optimizations
- **Attention Caching**: Fast focus computation and retrieval
- **Batch Processing**: Efficient tensor operations and memory pooling
- **Inference Caching**: Cached PLN results for repeated computations  
- **Memory Management**: Optimized AtomSpace access and working memory

### 🎯 Goal-Directed Behavior
- **Autonomous Goal Setting**: Dynamic goal creation and management
- **Achievement Tracking**: Success/failure monitoring with learning
- **Episodic Learning**: Experience-based decision making
- **Memory Consolidation**: Important experience transfer to long-term storage

### 🧪 Comprehensive Testing
- **Unit Tests**: Individual module validation
- **Integration Tests**: End-to-end cognitive architecture testing
- **Performance Tests**: Scalability and memory management validation
- **Error Tests**: Validation and error handling verification

## Files Overview

**Core Modules:**
- `OpenCogAtom.lua` - Knowledge units with attention and truth values
- `OpenCogAtomSpace.lua` - Hypergraph knowledge storage and retrieval  
- `OpenCogAttentionAllocation.lua` - Economic attention dynamics
- `OpenCogPLN.lua` - Probabilistic logic networks with advanced rules
- `OpenCogNetwork.lua` - Complete integrated cognitive architecture

**Enhanced Modules:**
- `OpenCogMetrics.lua` - Real-time cognitive performance monitoring
- `OpenCogWorkingMemory.lua` - Goal management and episodic memory
- `OpenCogValidator.lua` - Error handling and input validation utilities
- `OpenCogOptimizer.lua` - Performance optimization utilities

**Testing & Examples:**
- `test_opencog.lua` - Basic functionality tests
- `test_opencog_comprehensive.lua` - Complete test suite
- `examples/opencog_example.lua` - Usage demonstrations

## Quick Start Guide

```lua
require('nn')

-- Create cognitive architecture
local config = {
   atomSpaceCapacity = 1000,
   atomSize = 16, 
   perceptionSize = 10,
   actionSize = 8,
   focusSize = 50
}

local cogNet = nn.OpenCogNetwork(config)

-- Set goals
local goalEmb = torch.randn(16)
cogNet:setGoal("Learn patterns", goalEmb, 0.9)

-- Add knowledge
local catEmb = torch.randn(16)
cogNet:addKnowledge('Cat', catEmb, 
                   {sti=80, lti=0.7}, 
                   {strength=0.9, confidence=0.8})

-- Process perception and generate actions
local perception = torch.randn(1, 10) 
local actions = cogNet:forward(perception)

-- Advanced reasoning
local result = cogNet:performAdvancedInference({1, 2}, 
                                             {'deduction', 'similarity'})

-- Monitor cognitive state
local state = cogNet:getCognitiveState()
print("Attention efficiency:", state.metrics.attention.attentionalEfficiency)
print("Memory utilization:", state.metrics.memory.utilization)

-- Health check
local health = cogNet:healthCheck()
if health.overall then
   print("System healthy!")
else
   print("Issues detected:", table.concat(health.issues, ", "))
end
```

*This implementation brings the power of OpenCog's cognitive architecture to the neural network world, enabling the creation of truly intelligent systems that can reason, learn, and adapt.*