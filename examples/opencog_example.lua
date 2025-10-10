--[[
OpenCog Neural Network Example
=============================

This example demonstrates how to use the OpenCog cognitive architecture
modules for building AI systems with reasoning, attention, and memory.

Usage with Torch:
  th -lnn -e "dofile('examples/opencog_example.lua')"

This example shows:
1. Creating individual OpenCog components
2. Building a complete cognitive architecture  
3. Adding knowledge to the system
4. Performing reasoning and inference
5. Implementing perception-action loops
]]

require('nn')

print("OpenCog Neural Network Example")
print("==============================")

-- Configuration for the cognitive system
local config = {
   atomSpaceCapacity = 200,  -- Maximum atoms in memory
   atomSize = 12,            -- Dimensionality of atom embeddings
   focusSize = 20,           -- Size of attentional focus
   maxPremises = 3,          -- Maximum premises for PLN inference
   perceptionSize = 8,       -- Input perception dimensionality
   actionSize = 6,           -- Output action dimensionality
   cyclesPerForward = 3      -- Cognitive cycles per forward pass
}

print("Creating OpenCog cognitive architecture...")

-- Create the complete cognitive system
local cogNet = nn.OpenCogNetwork(config)

print("✓ Cognitive network created")
print("Initial state:", cogNet)

-- Example 1: Adding explicit knowledge to the system
print("\n--- Example 1: Knowledge Representation ---")

-- Create knowledge embeddings (normally these would come from perception)
local catEmbedding = torch.randn(config.atomSize)
local animalEmbedding = torch.randn(config.atomSize) 
local movementEmbedding = torch.randn(config.atomSize)

-- Add concepts with different importance and truth values
local catAtom = cogNet:addKnowledge('Cat', catEmbedding, 
                                   {sti = 80, lti = 0.7}, 
                                   {strength = 0.9, confidence = 0.8})

local animalAtom = cogNet:addKnowledge('Animal', animalEmbedding,
                                      {sti = 70, lti = 0.8},
                                      {strength = 0.95, confidence = 0.9})

local movesAtom = cogNet:addKnowledge('Moves', movementEmbedding,
                                     {sti = 60, lti = 0.5},
                                     {strength = 0.85, confidence = 0.7})

print("Added knowledge atoms:", catAtom, animalAtom, movesAtom)

-- Check knowledge base status
local kb = cogNet:getKnowledgeBase()
print("Knowledge base contains", kb.atomCount, "atoms")

-- Example 2: Performing logical reasoning
print("\n--- Example 2: PLN Reasoning ---")

-- Manual inference: If we have "Cat" and "Animal" concepts,
-- what can we infer about their relationship?
if kb.atomCount >= 2 then
   local inferenceResult = cogNet:performInference({catAtom, animalAtom}, 'deduction')
   print("Deductive inference completed")
   print("Conclusion truth value - Strength:", inferenceResult[1][config.atomSize + 1],
         "Confidence:", inferenceResult[1][config.atomSize + 2])
end

-- Example 3: Perception-Action Loop
print("\n--- Example 3: Perception-Action Processing ---")

-- Simulate sensory input (e.g., visual/auditory perception)
local perceptionInputs = torch.randn(3, config.perceptionSize)
print("Processing", perceptionInputs:size(1), "perception samples...")

-- Process through cognitive architecture
local actions = cogNet:forward(perceptionInputs)
print("Generated actions:", actions:size())
print("Sample action vector:", actions[1])

-- Example 4: Attention and Stimulation
print("\n--- Example 4: Attention Dynamics ---")

-- Apply external stimulation to certain concepts
-- (simulating reinforcement or important events)
cogNet:stimulate({
   [catAtom] = 15,      -- Strong positive stimulation for cat concept
   [animalAtom] = 10,   -- Moderate stimulation for animal concept
   [movesAtom] = -5     -- Negative stimulation for movement concept
})

print("Applied stimulation to atoms")

-- Check updated cognitive state
local cogState = cogNet:getCognitiveState()
print("Cognitive cycles completed:", cogNet.cycleCount)

-- Example 5: Learning from feedback
print("\n--- Example 5: Learning Integration ---")

-- Simulate a learning scenario where the system receives feedback
-- about its actions (this would normally come from environment rewards)

local criterion = nn.MSECriterion()
local targetActions = torch.randn(actions:size())  -- Desired actions

-- Compute loss and gradients
local loss = criterion:forward(actions, targetActions)
local gradOutput = criterion:backward(actions, targetActions)

print("Action loss:", loss)

-- Backpropagate through the cognitive architecture
local gradInput = cogNet:backward(perceptionInputs, gradOutput)
print("Gradients computed, shape:", gradInput:size())

-- In a real training setup, you would now update parameters using an optimizer
local params, gradParams = cogNet:parameters()
print("Total trainable parameters:", #params)

local totalParams = 0
for i, p in ipairs(params) do
   totalParams = totalParams + p:nElement()
end
print("Total parameter count:", totalParams)

-- Example 6: Monitoring System State  
print("\n--- Example 6: System Monitoring ---")

-- Get detailed knowledge base information
local finalKB = cogNet:getKnowledgeBase()
print("Final knowledge base:")
print("  - Total atoms:", finalKB.atomCount)
print("  - Capacity:", finalKB.capacity)
print("  - Cognitive cycles:", finalKB.cycleCount)
print("  - Focus size:", finalKB.attentionFocus:nElement())

-- Get economic balance of attention system
print("Attention allocation status:")
local balance = cogNet.attentionAllocation:economicBalance()
print("  - Total STI budget:", balance.totalSTI)
print("  - Total LTI budget:", balance.totalLTI)
print("  - Rent rate:", balance.rentRate)
print("  - Wage rate:", balance.wageRate)

-- Example 7: Advanced Reasoning Chains
print("\n--- Example 7: Complex Reasoning ---")

-- Create a reasoning chain: Cat → Animal → LivingThing
if kb.atomCount >= 3 then
   -- Add another concept
   local livingThingEmbedding = torch.randn(config.atomSize)
   local livingThingAtom = cogNet:addKnowledge('LivingThing', livingThingEmbedding,
                                              {sti = 75, lti = 0.9},
                                              {strength = 0.98, confidence = 0.95})
   
   -- Perform chained inference
   local chainResult = cogNet:performInference({catAtom, animalAtom, livingThingAtom}, 'deduction')
   print("Chained reasoning result obtained")
   
   -- The system has now learned: Cat → Animal → LivingThing
   -- This creates a conceptual hierarchy in the AtomSpace
end

print("\n--- Summary ---")
print("OpenCog cognitive architecture demonstration completed!")
print("The system now contains:")

local finalState = cogNet:getCognitiveState()
print("  ✓ Semantic knowledge in AtomSpace") 
print("  ✓ Attention allocation mechanisms")
print("  ✓ Probabilistic logic reasoning (PLN)")
print("  ✓ Perception-action integration")
print("  ✓ Learning and memory consolidation")
print("  ✓ Economic attention dynamics")

-- Example 8: Advanced Cognitive Features
print("\n--- Example 8: Advanced Cognitive Features ---")

-- Goal-directed behavior
print("Setting cognitive goals...")
local goalEmbedding = torch.randn(config.atomSize)
local goalId = cogNet:setGoal("Learn about cats", goalEmbedding, 0.9)
print("Goal set with ID:", goalId)

-- Advanced reasoning with multiple rules
local advancedInference = cogNet:performAdvancedInference({catAtom, animalAtom}, {'deduction', 'similarity'})
print("Advanced inference - Strength:", advancedInference[1], "Confidence:", advancedInference[2])

-- Prediction and episodic learning
local futureState = cogNet:predictNextState(perceptionInputs[1], 1)
print("Predicted future state size:", futureState:size())

-- Add episodic memories with rewards
cogNet:addEpisode(perceptionInputs[1], actions[1], 0.8)  -- positive experience
cogNet:addEpisode(perceptionInputs[2], actions[2], -0.2) -- negative experience
print("Episodic memories added")

-- Memory consolidation
cogNet:updateMemoryConsolidation()
print("Important memories consolidated to long-term storage")

-- Example 9: Cognitive Metrics and Monitoring
print("\n--- Example 9: Cognitive Metrics ---")

-- Get comprehensive cognitive metrics
local detailedState = cogNet:getCognitiveState()
print("Cognitive metrics collected:")
print("  - Attention efficiency:", string.format("%.3f", detailedState.metrics.attention.attentionalEfficiency))
print("  - Memory utilization:", string.format("%.3f", detailedState.metrics.memory.utilization))
print("  - Inference success rate:", string.format("%.3f", detailedState.metrics.reasoning.successRate))
print("  - Knowledge acquisition rate:", detailedState.metrics.learning.knowledgeAcquisitionRate)

-- Working memory status
print("Working memory status:")
print("  - Items in working memory:", detailedState.workingMemory.numItems)
print("  - Current goal:", detailedState.workingMemory.currentGoal)
print("  - Episodes stored:", detailedState.workingMemory.episodeCount)

print("\nThis cognitive architecture can be integrated into:")
print("  • Robotics systems for intelligent behavior")
print("  • Natural language processing with reasoning")
print("  • Game AI with strategic planning")
print("  • Knowledge graph learning and inference") 
print("  • Multi-modal AI systems")
print("  • Artificial General Intelligence research")
print("  • Autonomous decision-making systems")
print("  • Adaptive learning environments")

print("\nNew features in this enhanced version:")
print("  ✓ Advanced PLN inference rules (similarity, contraposition, etc.)")
print("  ✓ Working memory with goal stack and episodic buffer")
print("  ✓ Comprehensive cognitive metrics and monitoring") 
print("  ✓ Prediction and memory consolidation")
print("  ✓ Goal-directed reasoning and behavior")
print("  ✓ Real-time attention and memory analysis")

print("\nFor more examples, see: doc/opencog.md")