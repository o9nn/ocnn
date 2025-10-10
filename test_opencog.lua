#!/usr/bin/env lua

-- Test script for OpenCog neural network modules

require('torch')
require('nn')

print("Testing OpenCog Neural Network Modules")
print("=====================================")

-- Test OpenCogAtom
print("\n1. Testing OpenCogAtom...")
local atom = nn.OpenCogAtom(8, 'ConceptNode')
local input = torch.randn(2, 1):uniform(0.5, 1.0)  -- batch of activation values
local output = atom:forward(input)

print("Atom input size:", input:size())
print("Atom output size:", output:size())
print("Sample output:", output[1])
print("Atom info:", atom)

-- Test truth value operations
atom:setTruthValue(0.8, 0.9)
local tv = atom:getTruthValue()
print("Truth value set/get:", tv[1], tv[2])

-- Test attention value operations  
atom:updateSTI(10)
atom:updateLTI(0.1)
local av = atom:getAttentionValues()
print("Attention values:", av[1], av[2])

-- Test OpenCogAtomSpace
print("\n2. Testing OpenCogAtomSpace...")
local atomSpace = nn.OpenCogAtomSpace(100, 8)

-- Add some atoms
local atom1 = atomSpace:addAtom('ConceptNode', torch.randn(8), 80, 0.5, 0.8, 0.9)
local atom2 = atomSpace:addAtom('PredicateNode', torch.randn(8), 60, 0.3, 0.7, 0.8)
local atom3 = atomSpace:addAtom('LinkNode', torch.randn(8), 40, 0.2, 0.6, 0.7)

print("Added", atomSpace:getAtomCount(), "atoms to AtomSpace")
print("AtomSpace info:", atomSpace)

-- Test retrieval by indices
local indices = torch.LongTensor({1, 2})
local retrievedAtoms = atomSpace:forward(indices:view(-1, 1))
print("Retrieved atoms size:", retrievedAtoms:size())

-- Test attention focus
local focusIndices = atomSpace:getAttentionalFocus()
print("Attentional focus size:", focusIndices:nElement())

-- Test OpenCogAttentionAllocation
print("\n3. Testing OpenCogAttentionAllocation...")
local attention = nn.OpenCogAttentionAllocation(100, 20)

-- Create sample STI/LTI data
local stiLtiData = torch.Tensor(1, 3, 2)  -- batch=1, atoms=3, [STI,LTI]=2
stiLtiData[1][1] = torch.Tensor({80, 0.5})  -- atom 1: high STI, medium LTI
stiLtiData[1][2] = torch.Tensor({30, 0.2})  -- atom 2: low STI, low LTI
stiLtiData[1][3] = torch.Tensor({60, 0.8})  -- atom 3: medium STI, high LTI

print("Input attention data size:", stiLtiData:size())
local updatedAttention = attention:forward(stiLtiData)
print("Updated attention data size:", updatedAttention:size())
print("Attention module info:", attention)

-- Check economic balance
local balance = attention:economicBalance()
print("Economic balance - Total STI:", balance.totalSTI, "Total LTI:", balance.totalLTI)

-- Test OpenCogPLN
print("\n4. Testing OpenCogPLN...")  
local pln = nn.OpenCogPLN(2, 8)  -- max 2 premises, 8-dim conclusions

-- Create sample premises with embeddings and truth values
local premises = torch.Tensor(1, 2, 10)  -- batch=1, premises=2, embedding+strength+confidence=10
premises[1][1] = torch.cat(torch.randn(8), torch.Tensor({0.8, 0.9}))  -- premise 1
premises[1][2] = torch.cat(torch.randn(8), torch.Tensor({0.7, 0.8}))  -- premise 2

print("PLN input premises size:", premises:size())
local inference = pln:forward(premises)
print("PLN inference result size:", inference:size())
print("Conclusion truth value:", inference[1][9], inference[1][10])  -- strength, confidence
print("PLN info:", pln)

-- Test direct inference rules
local tv1 = {0.8, 0.9}
local tv2 = {0.7, 0.8}
local deduction_result = pln:deductionRule(tv1, tv2)
local induction_result = pln:inductionRule(tv1, tv2)
local revision_result = pln:revisionRule(tv1, tv2)

print("Deduction result:", deduction_result[1], deduction_result[2])
print("Induction result:", induction_result[1], induction_result[2])
print("Revision result:", revision_result[1], revision_result[2])

-- Test OpenCogNetwork (integrated system)
print("\n5. Testing OpenCogNetwork...")
local config = {
   atomSpaceCapacity = 50,
   atomSize = 8,
   focusSize = 10,
   maxPremises = 2,
   perceptionSize = 6,
   actionSize = 4
}

local cogNet = nn.OpenCogNetwork(config)
print("OpenCog Network created:", cogNet)

-- Test cognitive cycle
local perception = torch.randn(2, 6)  -- batch of 2 perception inputs
print("Perception input size:", perception:size())

local actions = cogNet:forward(perception)
print("Action output size:", actions:size())
print("Sample actions:", actions[1])

-- Add some knowledge
local knowledgeEmbedding = torch.randn(8)
local knowledgeAtom = cogNet:addKnowledge('TestConcept', knowledgeEmbedding, 
                                         {sti=70, lti=0.6}, {strength=0.9, confidence=0.8})
print("Added knowledge atom ID:", knowledgeAtom)

-- Get knowledge base status
local kb = cogNet:getKnowledgeBase()
print("Knowledge base - Atoms:", kb.atomCount, "Cycles:", kb.cycleCount)

-- Perform manual inference
if kb.atomCount >= 2 then
   local inferenceResult = cogNet:performInference({1, 2}, 'deduction')
   print("Manual inference result size:", inferenceResult:size())
end

-- Test stimulation
cogNet:stimulate({[1] = 10, [2] = 5})
print("Applied stimulation to atoms")

-- Get cognitive state
local state = cogNet:getCognitiveState()
print("Cognitive state acquired")

print("\n6. Testing gradient computation...")
-- Test backward pass
local criterion = nn.MSECriterion()
local target = torch.randn(actions:size())
local loss = criterion:forward(actions, target)
local gradOutput = criterion:backward(actions, target)

local gradInput = cogNet:backward(perception, gradOutput)
print("Loss:", loss)
print("Gradient input size:", gradInput:size())

-- Test parameters
local params, gradParams = cogNet:parameters()
print("Total parameters:", #params)
local totalParamCount = 0
for i, p in ipairs(params) do
   totalParamCount = totalParamCount + p:nElement()
end
print("Total parameter count:", totalParamCount)

print("\n7. Testing advanced features...")

-- Test goal setting
local goalEmbedding = torch.randn(8)
local goalId = cogNet:setGoal("Find food", goalEmbedding, 0.9)
print("Set goal with ID:", goalId)

-- Test episodic memory
local episodeId = cogNet:addEpisode(perception[1], actions[1], 0.7)
print("Added episode with ID:", episodeId)

-- Test advanced inference
local advancedResult = cogNet:performAdvancedInference({1, 2}, {'deduction', 'similarity'})
print("Advanced inference result:", advancedResult[1], advancedResult[2])

-- Test prediction
local prediction = cogNet:predictNextState(perception[1], 1)
print("Prediction generated, size:", prediction:size())

-- Test memory consolidation
cogNet:updateMemoryConsolidation()
print("Memory consolidation completed")

print("\n8. Testing cognitive metrics...")
-- Test metrics
local metricsModule = nn.OpenCogMetrics()
local cogState = cogNet:getCognitiveState()
local metricsOutput = metricsModule:forward(cogState.basic)
print("Metrics computed:", metricsOutput:size())
print("Detailed metrics available:", type(cogState.metrics))

print("\n9. Testing working memory...")
-- Test working memory
local workingMemory = nn.OpenCogWorkingMemory(10, 8)
local wmState = workingMemory:getWorkingMemoryState()
print("Working memory state:", wmState.numItems, "items")

print("\n✓ All OpenCog module tests completed successfully!")
print("Enhanced OpenCog neural network implementation is ready.")