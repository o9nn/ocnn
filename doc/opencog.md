# OpenCog Neural Network Implementation

This document describes the OpenCog cognitive architecture implementation as neural network modules within the Torch nn package.

## Overview

The OpenCog implementation provides a complete cognitive architecture as differentiable neural network modules, enabling the integration of symbolic reasoning, attention allocation, and knowledge representation within end-to-end trainable systems.

## Core Modules

### 1. OpenCogAtom (`nn.OpenCogAtom`)

Represents a basic unit of knowledge with importance values and truth values.

```lua
atom = nn.OpenCogAtom(atomSize, atomType)
```

**Parameters:**
- `atomSize` (number): Dimensionality of the atom embedding (default: 10)
- `atomType` (string): Type of atom - 'ConceptNode', 'PredicateNode', etc. (default: 'ConceptNode')

**Key Features:**
- Learnable embedding representation
- Short Term Importance (STI) and Long Term Importance (LTI) values
- Probabilistic truth values (strength and confidence)
- Differentiable attention and truth value updates

**Methods:**
- `atom:updateSTI(delta)` - Update Short Term Importance
- `atom:updateLTI(delta)` - Update Long Term Importance  
- `atom:getTruthValue()` - Get current {strength, confidence}
- `atom:setTruthValue(strength, confidence)` - Set truth values
- `atom:getAttentionValues()` - Get current {STI, LTI}

**Input/Output:**
- Input: `[batchSize, inputDim]` - activation/stimulation values
- Output: `[batchSize, atomSize + 4]` - embedding + STI + LTI + strength + confidence

### 2. OpenCogAtomSpace (`nn.OpenCogAtomSpace`)

Hypergraph database for storing and retrieving collections of atoms with attention mechanisms.

```lua
atomSpace = nn.OpenCogAtomSpace(capacity, atomSize)
```

**Parameters:**
- `capacity` (number): Maximum number of atoms (default: 1000)
- `atomSize` (number): Size of each atom embedding (default: 10)

**Key Features:**
- Dynamic atom storage with forgetting mechanism based on LTI
- Attention-weighted retrieval using softmax over STI values
- Support for multiple atom types
- Similarity-based query processing

**Methods:**
- `atomSpace:addAtom(atomType, embedding, sti, lti, strength, confidence)` - Add new atom
- `atomSpace:getAttentionalFocus()` - Get indices of high-attention atoms
- `atomSpace:stimulateAtom(atomIndex, stimulation)` - Apply external stimulation
- `atomSpace:forgetLowImportanceAtoms(numToForget)` - Remove low-LTI atoms
- `atomSpace:getAtomCount()` - Current number of atoms

**Input/Output:**
- Input: `[batchSize, 1]` (atom indices) or `[batchSize, queryDim]` (query vectors)
- Output: `[batchSize, atomSize + 6]` - embedding + STI + LTI + strength + confidence + attention + type

### 3. OpenCogAttentionAllocation (`nn.OpenCogAttentionAllocation`)

Manages STI and LTI dynamics through economic attention allocation mechanisms.

```lua
attention = nn.OpenCogAttentionAllocation(atomSpaceSize, focusSize)
```

**Parameters:**
- `atomSpaceSize` (number): Size of the atom space (default: 1000)
- `focusSize` (number): Size of attentional focus (default: 50)

**Key Features:**
- Economic model with rent collection and wage payment
- STI decay and LTI promotion mechanisms
- Attentional focus boundary management
- Hebbian learning integration

**Attention Dynamics:**
1. Collect rent from all atoms based on STI
2. Apply STI decay
3. Identify atoms in attentional focus (STI > threshold)
4. Pay wages to focused atoms
5. Promote high-STI atoms to higher LTI
6. Ensure STI conservation within budget

**Methods:**
- `attention:economicBalance()` - Get current economic parameters
- `attention:getAttentionalFocus(stiValues)` - Get focused atom indices
- `attention:stimulateAtoms(indices, amounts)` - Apply external stimulation

**Input/Output:**
- Input: `[batchSize, numAtoms, 2]` (STI/LTI) or `[batchSize, numAtoms, fullDim]`
- Output: Same format with updated attention values

### 4. OpenCogPLN (`nn.OpenCogPLN`)

Probabilistic Logic Networks for uncertain reasoning and inference.

```lua
pln = nn.OpenCogPLN(maxPremises, conclusionSize)
```

**Parameters:**
- `maxPremises` (number): Maximum premises per inference (default: 3)
- `conclusionSize` (number): Size of conclusion representation (default: 10)

**Key Features:**
- Differentiable implementation of PLN inference rules
- Support for deduction, induction, abduction, revision, and choice
- Learnable inference weights and truth value combinations
- Confidence-based belief integration

**Inference Rules:**
- **Deduction:** `(A → B, B → C) ⊢ (A → C)` 
- **Induction:** `(A → B, A) ⊢ B` (with reduced confidence)
- **Abduction:** `(B → C, A → B) ⊢ (A → C)` (with low confidence)
- **Revision:** Combine two beliefs about the same statement
- **Choice:** Select between conflicting beliefs

**Methods:**
- `pln:deductionRule(premise1_tv, premise2_tv)` - Apply deduction
- `pln:inductionRule(premise1_tv, premise2_tv)` - Apply induction
- `pln:abductionRule(premise1_tv, premise2_tv)` - Apply abduction
- `pln:revisionRule(belief1_tv, belief2_tv)` - Combine beliefs
- `pln:choiceRule(belief1_tv, belief2_tv)` - Choose best belief
- `pln:inferenceChain(premises, ruleSequence)` - Apply rule sequence

**Input/Output:**
- Input: `[batchSize, maxPremises, premiseSize]` - premises with embeddings + truth values
- Output: `[batchSize, conclusionSize + 2]` - conclusion embedding + strength + confidence

### 5. OpenCogNetwork (`nn.OpenCogNetwork`)

Complete integrated OpenCog cognitive architecture combining all components.

```lua
config = {
   atomSpaceCapacity = 1000,
   atomSize = 16,
   focusSize = 50,
   maxPremises = 3,
   perceptionSize = 10,
   actionSize = 10,
   cyclesPerForward = 5
}
cogNet = nn.OpenCogNetwork(config)
```

**Cognitive Cycle:**
1. **Perception Phase:** Process sensory input, create perception atoms
2. **Attention Allocation Phase:** Update STI/LTI values, manage focus
3. **Reasoning Phase:** Apply PLN inference to focused atoms  
4. **Action Selection Phase:** Generate actions based on reasoning + goals
5. **Learning Phase:** Update importance based on action outcomes

**Key Features:**
- End-to-end differentiable cognitive architecture
- Working memory and goal system
- Perception-to-action pipeline
- Knowledge acquisition and consolidation
- Cognitive state monitoring

**Methods:**
- `cogNet:addKnowledge(name, embedding, importance, truthValue)` - Add explicit knowledge
- `cogNet:performInference(premises, ruleType)` - Manual PLN inference
- `cogNet:stimulate(stimuli)` - Apply external atom stimulation
- `cogNet:getKnowledgeBase()` - Get AtomSpace status
- `cogNet:getCognitiveState()` - Get complete cognitive state

**Input/Output:**
- Input: `[batchSize, perceptionSize]` - sensory/perception data
- Output: `[batchSize, actionSize]` - action decisions/motor commands

## Usage Examples

### Basic Atom Operations
```lua
-- Create an atom
atom = nn.OpenCogAtom(8, 'ConceptNode')
input = torch.randn(2, 1)
output = atom:forward(input)

-- Set truth value
atom:setTruthValue(0.8, 0.9)
tv = atom:getTruthValue()

-- Update attention
atom:updateSTI(10)
atom:updateLTI(0.1)
```

### AtomSpace Knowledge Storage
```lua
-- Create AtomSpace
atomSpace = nn.OpenCogAtomSpace(100, 8)

-- Add atoms
atom1 = atomSpace:addAtom('ConceptNode', torch.randn(8), 80, 0.5, 0.8, 0.9)
atom2 = atomSpace:addAtom('PredicateNode', torch.randn(8), 60, 0.3, 0.7, 0.8)

-- Retrieve atoms
indices = torch.LongTensor({1, 2})
atoms = atomSpace:forward(indices:view(-1, 1))

-- Get attentional focus
focus = atomSpace:getAttentionalFocus()
```

### PLN Reasoning
```lua  
-- Create PLN module
pln = nn.OpenCogPLN(2, 8)

-- Prepare premises
premises = torch.Tensor(1, 2, 10)
premises[1][1] = torch.cat(torch.randn(8), torch.Tensor({0.8, 0.9}))
premises[1][2] = torch.cat(torch.randn(8), torch.Tensor({0.7, 0.8}))

-- Perform inference
conclusion = pln:forward(premises)

-- Manual rule application
tv1 = {0.8, 0.9}
tv2 = {0.7, 0.8}
result = pln:deductionRule(tv1, tv2)
```

### Complete Cognitive System
```lua
-- Configure and create network
config = {atomSpaceCapacity=50, atomSize=8, perceptionSize=6, actionSize=4}
cogNet = nn.OpenCogNetwork(config)

-- Process perception and generate actions
perception = torch.randn(2, 6)
actions = cogNet:forward(perception)

-- Add knowledge
embedding = torch.randn(8)
atomId = cogNet:addKnowledge('TestConcept', embedding, {sti=70, lti=0.6}, {strength=0.9, confidence=0.8})

-- Manual inference
result = cogNet:performInference({1, 2}, 'deduction')

-- Monitor system
state = cogNet:getCognitiveState()
kb = cogNet:getKnowledgeBase()
```

## Training and Integration

The OpenCog modules are fully compatible with standard PyTorch/Torch training procedures:

```lua
-- Standard training loop
criterion = nn.MSECriterion()
optimizer = optim.sgd

for epoch = 1, numEpochs do
   for batch in dataloader do
      local input, target = batch.input, batch.target
      
      -- Forward pass
      local output = cogNet:forward(input)
      local loss = criterion:forward(output, target)
      
      -- Backward pass
      local gradOutput = criterion:backward(output, target)
      cogNet:backward(input, gradOutput)
      
      -- Update parameters
      optimizer(function() return loss end, params, learningRate)
   end
end
```

## Implementation Notes

1. **Memory Management:** The AtomSpace implements forgetting based on LTI values to maintain bounded memory usage.

2. **Attention Dynamics:** The attention allocation follows OpenCog's economic model with rent, wages, and STI decay.

3. **PLN Integration:** Truth values are maintained throughout the reasoning process with proper confidence propagation.

4. **Differentiability:** All operations are implemented to maintain gradient flow for end-to-end training.

5. **Cognitive Cycles:** The network can be configured to run multiple cognitive cycles per forward pass for deeper processing.

## Applications

This OpenCog implementation enables:

- **Cognitive AI Systems:** Full AGI architectures with reasoning, learning, and memory
- **Hybrid Neuro-Symbolic Models:** Combining deep learning with symbolic reasoning
- **Attention-based Learning:** Dynamic focus allocation for multi-task learning
- **Knowledge Graph Processing:** Differentiable graph reasoning and inference
- **Cognitive Robotics:** Perception-action loops with embedded reasoning

## References

- Goertzel, B. et al. "OpenCog: A Software Framework for Integrative Artificial General Intelligence"
- Goertzel, B. "Probabilistic Logic Networks: A Comprehensive Framework for Uncertain Inference"
- OpenCog Foundation documentation and specifications